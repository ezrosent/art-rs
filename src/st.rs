//! Single-threaded radix tree implementation based on HyPer's ART
extern crate smallvec;
extern crate stdsimd;

use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use self::node_variants::*;
use self::smallvec::{Array, SmallVec};
use self::stdsimd::vendor;
use self::stdsimd::simd;
use super::Digital;

pub trait Element {
    type Key: for<'a> Digital<'a>;
    fn key(&self) -> &Self::Key;
    fn matches(&self, k: &Self::Key) -> bool;
    fn replace_matching(&mut self, other: &mut Self);
}

type RawMutRef<'a, T> = &'a mut RawNode<T>;
type RawRef<'a, T> = &'a RawNode<T>;

pub struct RawART<T: Element> {
    len: usize,
    root: ChildPtr<T>,
}

macro_rules! with_node_inner {
    ($base_node:expr, $nod:ident, $body:expr, $r:tt) => {
        with_node_inner!($base_node, $nod, $body, $r, _)
    };
    ($base_node:expr, $nod:ident, $body:expr, $r:tt, $ty:tt) => {
        {
            let _b: $r<()> = $base_node;
            match _b.typ {
                NODE_4 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<()>, $r<Node4<$ty>>>(_b) };
                    $body
                },
                NODE_16 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<()>, $r<Node16<$ty>>>(_b) };
                    $body
                },
                NODE_48 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<()>, $r<Node48<$ty>>>(_b) };
                    $body
                },
                NODE_256 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<()>, $r<Node256<$ty>>>(_b) };
                    $body
                },
                _ => unreachable!(),
            }
        }
    };
}

macro_rules! with_node_mut {
    ($base_node:expr, $nod:ident, $body:expr) => {
        with_node_mut!($base_node, $nod, $body, _)
    };
    ($base_node:expr, $nod:ident, $body:expr, $ty:tt) => {
        with_node_inner!($base_node, $nod, $body, RawMutRef, $ty)
    };
}

macro_rules! with_node {
    ($base_node:expr, $nod:ident, $body:expr) => {
        with_node!($base_node, $nod, $body, _)
    };
    ($base_node:expr, $nod:ident, $body:expr, $ty:tt) => {
        with_node_inner!($base_node, $nod, $body, RawRef, $ty)
    };
}

impl<T: Element> RawART<T> {
    // TODO add support for skipping the `matches` check when no optimistic prefixes are
    // encountered (this should happen e.g. for all integer keys)

    pub fn new() -> Self {
        RawART {
            len: 0,
            root: ChildPtr::null(),
        }
    }

    // replace with NonNull
    pub fn lookup_raw(&mut self, k: &T::Key) -> Option<*mut T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        fn lookup_raw_recursive<T: Element>(
            curr: &mut ChildPtr<T>,
            k: &T::Key,
            digits: &[u8],
        ) -> Option<*mut T> {
            // TODO: If Rust ever support proper tail-calls, this could be made tail-recursive.
            // In lieu of that, it's worth profiling this code to determine if an ugly iterative
            // rewrite would be worthwhile.
            // TODO: take consumed prefix into account?
            match unsafe { curr.get_mut() } {
                None => None,
                Some(Ok(leaf_node)) => {
                    if leaf_node.matches(k) {
                        Some(leaf_node)
                    } else {
                        None
                    }
                }
                Some(Err(inner_node)) => {
                    // handle prefixes now
                    inner_node
                        .prefix_matches_optimistic(digits)
                        .and_then(|consumed| {
                            let new_digits = &digits[consumed..];
                            if new_digits.len() == 0 {
                                // Our digits were entirely consumed, but this is a non-leaf node.
                                // That means our node is not present.
                                return None;
                            }
                            with_node_mut!(inner_node, nod, {
                                nod.find_raw(new_digits[0]).and_then(|next_node| {
                                    lookup_raw_recursive(
                                        unsafe { &mut *next_node },
                                        k,
                                        &new_digits[1..],
                                    )
                                })
                            })
                        })
                }
            }
        }
        lookup_raw_recursive(&mut self.root, k, digits.as_slice())
    }
    pub unsafe fn delete_raw(&mut self, k: &T::Key) -> Option<*mut T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        unsafe fn delete_raw_recursive<T: Element>(
            k: &T::Key,
            curr: &mut ChildPtr<T>,
            parent: Option<(u8, &mut ChildPtr<T>)>,
            digits: &[u8],
        ) -> Option<*mut T> {
            if curr.is_null() {
                return None;
            }
            // TODO add delete method to Node, which should take in a digit and return either:
            //     Failure
            //     Success(unit)
            //     Success(childptr)
            // where the third case indicates that childptr is (was -- it must be moved out) the final child of the node,
            // in which case need to install that child into the parent node. For now, don't
            // attempt to increase the prefix.
            unimplemented!()
            // match curr.get_mut().unwrap() {
            //     Ok(leaf_node) => {
            //         if leaf_node.matches(k) {
            //             unimplemented!()
            //         }
            //     }
            // }
        }
        delete_raw_recursive(k, &mut self.root, None, &digits[..])
    }

    pub unsafe fn insert_raw(&mut self, elt: T) -> Result<(), T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(elt.key().digits());
        // want to do something similar to lookup_raw_recursive.
        // Need to keep track of:
        // - current node
        // - pointer to current node from parent
        // - elt
        // - digits
        unsafe fn insert_raw_recursive<T: Element>(
            curr: &mut ChildPtr<T>,
            mut e: T,
            digits: &[u8],
            mut consumed: usize,
        ) -> Result<(), T> {
            debug_assert!(consumed <= digits.len());
            if curr.is_null() {
                let new_leaf = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                *curr = new_leaf;
                return Ok(());
            }
            let leaf_digits = match curr.get_mut().unwrap() {
                Ok(leaf_node) => {
                    if leaf_node.matches(e.key()) {
                        // Found a matching leaf node. We swap in our value and return the old one.
                        leaf_node.replace_matching(&mut e);
                        return Err(e);
                    }
                    // found a leaf node, need to split it to a Node4 with two leaves
                    let mut leaf_digits = SmallVec::<[u8; 32]>::new();
                    leaf_digits.extend(leaf_node.key().digits());
                    debug_assert!(leaf_digits.len() <= digits.len());
                    leaf_digits
                }
                Err(inner_node) => {
                    // found an inner node. need to continue the search!
                    match inner_node.prefix_matches_pessimistic(&digits[consumed..]) {
                        Some(matched) => {
                            // TODO review the base case here
                            // This inner node shares a prefix with the current node. We recur into
                            // this node.
                            consumed += matched;
                            // XXX what if consumed == digits.len()? the structure of the keys must
                            // guarantee that we do not see this. For example, if we store u64s,
                            // then all keys are 8 bytes long so `consumed` cannot be more than 7.
                            //
                            // For variable-length keys, (like strings) we require a "stop"
                            // character to appear to avoid this problem. For us, the
                            // null-terminator is such a stop character.
                            debug_assert!(consumed < digits.len());
                            let d = digits[consumed];

                            with_node_mut!(inner_node, nod, {
                                // TODO validate the prefix logic here:
                                // if there's an optimistic prefix we may have to adjust its
                                // length...  for now it may be safer to just truncate the prefix
                                inner_node.count = cmp::min(inner_node.count, PREFIX_LEN as u32);
                                if let Some(next_ptr) = nod.find_mut(d) {
                                    return insert_raw_recursive(next_ptr, e, digits, consumed);
                                }
                                let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                                RawNode::insert(&mut (nod as *mut _), d, c_ptr);
                                return Ok(());
                            });
                        }
                        None => {
                            // Our inner node X shares a non-matching prefix with the current node
                            fn adjust_prefix<R, T: Element>(
                                n: &mut RawNode<R>,
                                by: usize,
                                leaf: Option<&T>,
                                consumed: usize,
                            ) {
                                let old_count = n.count as usize;
                                n.count -= by as u32;
                                let start: *const _ = &n.prefix[by];
                                // first we want to copy everything over
                                unsafe {
                                    ptr::copy(start, &mut n.prefix[0], PREFIX_LEN - by);
                                }
                                if old_count > PREFIX_LEN {
                                    let leaf_ref = leaf.unwrap();
                                    let needed = cmp::min(old_count - PREFIX_LEN, by);
                                    let mut i = 0;
                                    for (p, d) in n.prefix[PREFIX_LEN - by..]
                                        .iter_mut()
                                        .zip(leaf_ref.key().digits().skip(consumed))
                                    {
                                        *p = d;
                                    }
                                }
                            }
                            //
                            // There are X cases:
                            // 1. The two nodes stop overlapping at a *known* index
                            //    In this case, we make a new node with this known prefix,
                            //    we insert the new leaf into this new node and we insert the old
                            //    interior node into this new node as well, both at the digit where
                            //    they diverge. Next we have to readjust the prefix
                            //
                            // currently we only cut off PREFIX_LEN levels. We could do more, but
                            // that would involve getting
                            let mut common_prefix_digits = SmallVec::<[u8; 32]>::new();
                            let min_ref: Option<&T> = get_matching_prefix_node(
                                digits,
                                inner_node,
                                consumed,
                                &mut common_prefix_digits,
                                PhantomData,
                            );
                            let n4: Box<RawNode<Node4<T>>> =
                                make_node_with_prefix(&common_prefix_digits[..]);
                            debug_assert_eq!(n4.count, common_prefix_digits.len());
                            consumed += n4.count as usize;
                            adjust_prefix(
                                inner_node,
                                inner_node.count as usize - common_prefix_digits.len(),
                                min_ref,
                                consumed,
                            );
                            let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                            let n4_raw = Box::into_raw(n4);
                            // XXX: here we know that n4_raw will not get upgraded, that's the only
                            // reason re-using it here is safe.
                            RawNode::insert(&mut n4_raw, digits[consumed], c_ptr);

                            // We want to insert curr into n4, that means we need to figure out
                            // what its initial digit should be.
                            //
                            // We know that inner_node has some nonempty prefix (if it has an empty
                            // prefix we would have matched it). What we want is the first
                            // non-matching node in the prefix.
                            debug_assert!(inner_node.count > 0);
                            RawNode::insert(&mut n4_raw, inner_node.prefix[0], *curr);
                            *curr = ChildPtr::from_node(n4_raw);
                            return Ok(());
                        }
                    }
                }
            };
            let n4: Box<RawNode<Node4<T>>> = make_node_from_common_prefix(
                &leaf_digits.as_slice()[consumed..],
                &digits[consumed..],
            );
            let prefix_len = n4.count as usize;
            let mut n4_raw = Box::into_raw(n4);
            let mut leaf_ptr = ChildPtr::from_node(n4_raw);
            let new_leaf = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
            mem::swap(curr, &mut leaf_ptr);
            RawNode::insert(
                &mut n4_raw,
                // XXX seems wrong
                leaf_digits[consumed + prefix_len - 1],
                leaf_ptr,
            );
            RawNode::insert(&mut n4_raw, digits[consumed + prefix_len - 1], new_leaf);
            Ok(())
        }
        insert_raw_recursive(&mut self.root, elt, digits.as_slice(), 0)
    }
}

// may want to get the algorithmics right first. Just assume you're storing usize and have
// Either<Leaf, Option<Box<Node...>>> as the child pointers.
// TODO: use NonNull everywhere

struct ChildPtr<T>(usize, PhantomData<T>);

impl<T> Drop for ChildPtr<T> {
    fn drop(&mut self) {
        unsafe {
            match self.get_mut() {
                None => return,
                Some(Ok(x)) => mem::drop(Box::from_raw(x)),
                // with_node_mut! will "un-erase" the actual type of the RawNode. We want to call drop
                // on that to ensure all children are dropped
                // ... and to avoid undefined behavior, as we could wind up passing free the wrong size :)
                Some(Err(x)) => with_node_mut!(x, nod, mem::drop(Box::from_raw(nod)), T),
            }
        }
    }
}

impl<T> ChildPtr<T> {
    fn null() -> Self {
        ChildPtr(0, PhantomData)
    }

    fn from_node<R>(p: *mut RawNode<R>) -> Self {
        debug_assert!(!p.is_null());
        ChildPtr(p as usize, PhantomData)
    }

    fn from_leaf(p: *mut T) -> Self {
        debug_assert!(!p.is_null());
        ChildPtr((p as usize) & 1, PhantomData)
    }

    fn is_null(&self) -> bool {
        self.0 == 0
    }

    unsafe fn get(&self) -> Option<Result<&T, &RawNode<()>>> {
        if self.0 == 0 {
            None
        } else if self.0 & 1 == 1 {
            Some(Ok(&*((self.0 & !1) as *const T)))
        } else {
            Some(Err(&*(self.0 as *const RawNode<()>)))
        }
    }

    unsafe fn get_mut(&mut self) -> Option<Result<&mut T, &mut RawNode<()>>> {
        if self.0 == 0 {
            None
        } else if self.0 & 1 == 1 {
            Some(Ok(&mut *((self.0 & !1) as *mut T)))
        } else {
            Some(Err(&mut *(self.0 as *mut RawNode<()>)))
        }
    }
}

unsafe fn place_in_hole_at<T>(slice: &mut [T], at: usize, v: T, buff_len: usize) {
    let raw_p = slice.get_unchecked_mut(0) as *mut T;
    let target = raw_p.offset(at as isize);
    ptr::copy(target, raw_p.offset(at as isize + 1), buff_len - at - 1);
    ptr::write(target, v);
}

#[cfg(test)]
mod place_test {
    use super::*;

    #[test]
    fn place_in_hole_test() {
        let mut v1 = vec![0, 1, 3, 4, 0];
        let len = v1.len();
        unsafe {
            place_in_hole_at(&mut v1[..], 2, 2, len);
        }
        assert_eq!(v1, vec![0, 1, 2, 3, 4]);
    }
}

const PREFIX_LEN: usize = 8;

#[repr(C)]
struct RawNode<Footer> {
    typ: NodeType,
    children: u16,
    count: u32,
    prefix: [u8; PREFIX_LEN],
    node: Footer,
}

impl<T> RawNode<T> {
    fn prefix_matches_optimistic(&self, digits: &[u8]) -> Option<usize> {
        let count = self.count as usize;
        if digits.len() < count {
            return None;
        }
        for i in 0..cmp::min(count, PREFIX_LEN) {
            if digits[i] != self.prefix[i] {
                return None;
            }
        }
        Some(count)
    }

    fn prefix_matches_pessimistic(&self, digits: &[u8]) -> Option<usize> {
        let count = cmp::min(self.count as usize, PREFIX_LEN);
        if digits.len() < count {
            return None;
        }
        for i in 0..count {
            if digits[i] != self.prefix[i] {
                return None;
            }
        }

        Some(count)
    }
}

trait Node<T>: Sized {
    // insert assumes that 'd' is not present here already
    unsafe fn insert(curr: &mut *mut Self, d: u8, ptr: ChildPtr<T>);
    // TODO add a `delete` method
    fn get_min(&self) -> Option<&T>;
    fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>>;
    fn find(&self, d: u8) -> Option<&ChildPtr<T>> {
        self.find_raw(d).map(|raw_ptr| unsafe { &*raw_ptr })
    }
    fn find_mut(&self, d: u8) -> Option<&mut ChildPtr<T>> {
        self.find_raw(d).map(|raw_ptr| unsafe { &mut *raw_ptr })
    }
}

fn get_matching_prefix_slice<'a, 'b, A, I1, I2>(d1: I1, d2: I2, v: &mut SmallVec<A>)
where
    A: Array<Item = u8>,
    I1: Iterator<Item = &'a u8>,
    I2: Iterator<Item = &'b u8>,
{
    for (d1, d2) in d1.zip(d2) {
        if *d1 != *d2 {
            return;
        }
        v.push(*d1)
    }
}

fn get_matching_prefix_into<'a, 'b, A, I1, I2>(d1: I1, d2: I2, v: &mut SmallVec<A>)
where
    A: Array<Item = u8>,
    I1: Iterator<Item = &'a u8>,
    I2: Iterator<Item = u8>,
{
    for (d1, d2) in d1.zip(d2) {
        if *d1 != d2 {
            return;
        }
        v.push(*d1)
    }
}

fn get_matching_prefix_node<'a, T: Element, A: Array<Item = u8>>(
    d1: &[u8],
    n: &'a RawNode<()>,
    consumed: usize,
    v: &mut SmallVec<A>,
    _marker: PhantomData<T>,
) -> Option<&'a T> {
    // XXX: may want to specialize implementations of `skip` and return an iterator here watch this
    debug_assert_eq!(v.len(), 0);
    let node_explicit_prefix_len = cmp::min(PREFIX_LEN, n.count as usize);
    get_matching_prefix_slice(
        d1[consumed..].iter(),
        n.prefix[..node_explicit_prefix_len].iter(),
        v,
    );
    if v.len() < d1.len() && v.len() == PREFIX_LEN {
        with_node!(
            n,
            node,
            {
                let min_elt = node.get_min().expect("prefix node is empty!");
                let new_consumed = consumed + v.len();
                get_matching_prefix_into(
                    d1[new_consumed..].iter(),
                    min_elt.key().digits().skip(consumed),
                    v,
                );
                Some(min_elt)
            },
            T
        )
    } else {
        None
    }
}

fn make_node_with_prefix<T>(prefix: &[u8]) -> Box<RawNode<Node4<T>>> {
    let mut new_node = Box::new(RawNode {
        typ: NODE_4,
        children: 0,
        count: prefix.len() as u32,
        prefix: [0; PREFIX_LEN],
        node: Node4 {
            keys: [0; 4],
            ptrs: unsafe { mem::transmute::<[usize; 4], [ChildPtr<T>; 4]>([0 as usize; 4]) },
        },
    });
    let new_len = cmp::min(prefix.len(), PREFIX_LEN);
    unsafe {
        ptr::copy_nonoverlapping(
            &prefix[0] as *const _,
            &mut new_node.prefix[0] as *mut _,
            new_len,
        );
    }
    new_node
}

fn make_node_from_common_prefix<T>(d1: &[u8], d2: &[u8]) -> Box<RawNode<Node4<T>>> {
    let mut common_prefix_digits = SmallVec::<[u8; 32]>::new();
    get_matching_prefix_slice(d1.iter(), d2.iter(), &mut common_prefix_digits);
    make_node_with_prefix(&common_prefix_digits[..])
}

struct Node4<T> {
    keys: [u8; 4],
    ptrs: [ChildPtr<T>; 4],
}

mod node_variants {
    use super::*;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NodeType(u16);
    pub const NODE_4: NodeType = NodeType(1);
    pub const NODE_16: NodeType = NodeType(2);
    pub const NODE_48: NodeType = NodeType(3);
    pub const NODE_256: NodeType = NodeType(4);

    impl<T> Node<T> for RawNode<Node4<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            debug_assert!(self.children <= 4);
            for i in 0..4 {
                if self.node.keys[i] == d {
                    unsafe { return Some(self.node.ptrs.get_unchecked(i) as *const _ as *mut _) };
                }
                if i == self.children as usize {
                    return None;
                }
            }
            unreachable!()
        }

        fn get_min(&self) -> Option<&T> {
            debug_assert!(self.children <= 4);
            if self.children == 0 {
                return None;
            }
            let min_key_ix = 0; // we keep the child list sorted
            match unsafe { self.node.ptrs[min_key_ix as usize].get().unwrap() } {
                Ok(t) => Some(t),
                Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
            }
        }

        unsafe fn insert(curr: &mut *mut RawNode<Node4<T>>, d: u8, ptr: ChildPtr<T>) {
            debug_assert_eq!((**curr).typ, NODE_4);
            let slf: &mut RawNode<Node4<T>> = &mut **curr;
            if slf.children == 4 {
                let new_node = &mut *Box::into_raw(Box::new(RawNode {
                    typ: NODE_16,
                    children: slf.children,
                    count: slf.count,
                    prefix: slf.prefix,
                    node: Node16 {
                        keys: [0; 16],
                        ptrs: mem::transmute::<[usize; 16], [ChildPtr<T>; 16]>([0 as usize; 16]),
                    },
                }));
                ptr::swap_nonoverlapping(&mut slf.node.keys[0], &mut new_node.node.keys[0], 4);
                ptr::swap_nonoverlapping(&mut slf.node.ptrs[0], &mut new_node.node.ptrs[0], 4);
                let new_node_raw = new_node as *mut _ as *mut RawNode<Node4<T>>;
                *curr = new_node_raw;
                // free the old node
                mem::drop(Box::from_raw(slf));
                // make the recursive call
                return RawNode::<Node16<T>>::insert(mem::transmute(curr), d, ptr);
            }
            for i in 0..4 {
                if i == (slf.children as usize) {
                    // found an empty slot!
                    debug_assert_eq!(slf.node.keys[i], 0);
                    slf.node.keys[i] = d;
                    slf.node.ptrs[i] = ptr;
                    slf.children += 1;
                    return;
                }
                let cur_digit = slf.node.keys[i];
                debug_assert!(cur_digit != d);
                if cur_digit > d {
                    // we keep the list sorted, so we need to move all other entries ahead by 1
                    // slot. This is not safe if it's already full.
                    if slf.children == 4 {
                        break;
                    }
                    place_in_hole_at(&mut slf.node.keys[..], i, d, 4);
                    place_in_hole_at(&mut slf.node.ptrs[..], i, ptr, 4);
                    slf.children += 1;
                    return;
                }
            }
        }
    }

    pub struct Node16<T> {
        keys: [u8; 16],
        ptrs: [ChildPtr<T>; 16],
    }

    impl<T> Node<T> for RawNode<Node16<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            let mask = (1 << (self.children as usize)) - 1;
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
            {
                let ks = simd::u8x16::load(&self.node.keys[..], 0);
                let d_splat = simd::u8x16::splat(d);
                let comps = d_splat.eq(ks);
                let bits = unsafe { vendor::_mm_movemask_epi8(mem::transmute(comps)) } & mask;
                return if bits == 0 {
                    None
                } else {
                    debug_assert_eq!(bits.count_ones(), 1);
                    let target = bits.trailing_zeros();
                    debug_assert!(target < 16);
                    Some(unsafe {
                        self.node.ptrs.get_unchecked(target as usize) as *const _ as *mut _
                    })
                };
            }
            #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"),
                          target_feature = "sse2")))]
            {
                // copy over Node4 implementation
                unimplemented!()
            }
        }

        fn get_min(&self) -> Option<&T> {
            debug_assert!(self.children <= 16);
            if self.children == 0 {
                return None;
            }
            let min_key_ix = 0; // we keep the child list sorted
            match unsafe { self.node.ptrs[min_key_ix as usize].get().unwrap() } {
                Ok(t) => Some(t),
                Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
            }
        }

        unsafe fn insert(curr: &mut *mut RawNode<Node16<T>>, d: u8, ptr: ChildPtr<T>) {
            debug_assert_eq!((**curr).typ, NODE_16);
            let slf: &mut RawNode<Node16<T>> = &mut **curr;
            let mask = (1 << (slf.children as usize)) - 1;
            if slf.children == 16 {
                // upgrade
                let new_node = &mut *Box::into_raw(Box::new(RawNode {
                    typ: NODE_48,
                    children: 16,
                    count: slf.count,
                    prefix: slf.prefix,
                    node: Node48 {
                        keys: [0; 256],
                        ptrs: mem::transmute::<[usize; 48], [ChildPtr<T>; 48]>([0 as usize; 48]),
                    },
                }));
                for i in 0..16 {
                    let ix = slf.node.keys[i] as usize;
                    mem::swap(
                        slf.node.ptrs.get_unchecked_mut(ix),
                        new_node.node.ptrs.get_unchecked_mut(i),
                    );
                }
                let rev_cur = curr as *mut _ as *mut *mut RawNode<Node48<T>>;
                *rev_cur = new_node;
                mem::drop(Box::from_raw(slf));
                RawNode::<Node48<T>>::insert(&mut *rev_cur, d, ptr);
                return;
            }
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
            {
                let ks = simd::u8x16::load(&slf.node.keys[..], 0);
                let d_splat = simd::u8x16::splat(d);
                let comps = d_splat.lt(ks);
                let bits = vendor::_mm_movemask_epi8(mem::transmute(comps)) & mask;
                let target = bits.trailing_zeros() as usize;
                place_in_hole_at(&mut slf.node.keys[..], target, d, 16);
                place_in_hole_at(&mut slf.node.ptrs[..], target, ptr, 16);
            }
            #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"),
                          target_feature = "sse2")))]
            {
                // copy over Node16 implementation
                unimplemented!()
            }
            slf.children += 1;
        }
    }

    pub struct Node48<T> {
        keys: [u8; 256],
        ptrs: [ChildPtr<T>; 48],
    }

    impl<T> Node<T> for RawNode<Node48<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            let ix = unsafe { *self.node.keys.get_unchecked(d as usize) as usize };
            if ix == 0 {
                None
            } else {
                unsafe {
                    Some(self.node.ptrs.get_unchecked(ix - 1) as *const _ as *mut ChildPtr<T>)
                }
            }
        }

        fn get_min(&self) -> Option<&T> {
            const KEYS_PER_WORD: usize = 8;
            const N_WORDS: usize = 256 / KEYS_PER_WORD;
            if self.children == 0 {
                return None;
            }
            let keys_words =
                unsafe { mem::transmute::<&[u8; 256], &[u64; N_WORDS]>(&self.node.keys) };
            for i in 0..N_WORDS {
                let word = keys_words[i];
                if word == 0 {
                    continue;
                }
                let word_bytes = unsafe { mem::transmute::<u64, [u8; 8]>(word) };
                for b in &word_bytes[..] {
                    if *b != 0 {
                        let ix = *b - 1;
                        return match unsafe { self.node.ptrs[ix as usize].get().unwrap() } {
                            Ok(t) => Some(t),
                            Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
                        };
                    }
                }
            }
            unreachable!()
        }

        unsafe fn insert(curr: &mut *mut RawNode<Node48<T>>, d: u8, ptr: ChildPtr<T>) {
            debug_assert_eq!((**curr).typ, NODE_48);
            let slf: &mut RawNode<Node48<T>> = &mut **curr;
            debug_assert!(slf.children <= 48);
            if slf.children == 48 {
                let new_node = &mut *Box::into_raw(Box::new(RawNode {
                    typ: NODE_256,
                    children: 48,
                    count: slf.count,
                    prefix: slf.prefix,
                    node: Node256 {
                        ptrs: mem::transmute::<[usize; 256], [ChildPtr<T>; 256]>([0 as usize; 256]),
                    },
                }));
                for i in 0..256 {
                    let ix = *slf.node.keys.get_unchecked(i) as usize;
                    mem::swap(
                        slf.node.ptrs.get_unchecked_mut(ix),
                        new_node.node.ptrs.get_unchecked_mut(i),
                    );
                }
                let rev_cur = curr as *mut _ as *mut *mut RawNode<Node256<T>>;
                *rev_cur = new_node;
                mem::drop(Box::from_raw(slf));
                RawNode::<Node256<T>>::insert(&mut *rev_cur, d, ptr);
                return;
            }
            for i in 0..48 {
                let slot = slf.node.ptrs.get_unchecked_mut(i);
                if slot.is_null() {
                    ptr::write(slot, ptr);
                    slf.node.keys[d as usize] = i as u8 + 1;
                    slf.children += 1;
                    return;
                }
            }
            unreachable!()
        }
    }

    pub struct Node256<T> {
        ptrs: [ChildPtr<T>; 256],
    }

    impl<T> Node<T> for RawNode<Node256<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            unsafe {
                let p: *mut ChildPtr<T> =
                    self.node.ptrs.get_unchecked(d as usize) as *const _ as *mut _;
                if p.is_null() {
                    None
                } else {
                    Some(p)
                }
            }
        }

        fn get_min(&self) -> Option<&T> {
            // TODO benchmark with this vs. 0..256 + get_unchecked.
            // This search can also be sped up using simd to bulk-compare >0 for each cell.
            // While probably overkill, worth exploring if this sort of search becomes expensive.
            if self.children == 0 {
                return None;
            }

            for p in &self.node.ptrs[..] {
                if p.is_null() {
                    continue;
                }
                return match unsafe { p.get().unwrap() } {
                    Ok(t) => Some(t),
                    Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
                };
            }
            unreachable!()
        }

        unsafe fn insert(curr: &mut *mut RawNode<Node256<T>>, d: u8, ptr: ChildPtr<T>) {
            debug_assert_eq!((**curr).typ, NODE_256);
            let slf: &mut RawNode<Node256<T>> = &mut **curr;
            debug_assert!(slf.children <= 256);
            ptr::write(slf.node.ptrs.get_unchecked_mut(d as usize), ptr);
        }
    }
}
