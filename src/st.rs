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

pub struct ArtElement<T: for<'a> Digital<'a> + PartialEq>(T);

impl<T: for<'a> Digital<'a> + PartialEq> ArtElement<T> {
    pub fn new(t: T) -> ArtElement<T> {
        ArtElement(t)
    }
}

impl<T: for<'a> Digital<'a> + PartialEq> Element for ArtElement<T> {
    type Key = T;
    fn key(&self) -> &T {
        &self.0
    }

    fn matches(&self, k: &Self::Key) -> bool {
        *k == self.0
    }

    fn replace_matching(&mut self, other: &mut ArtElement<T>) {
        debug_assert!(self.matches(other.key()));
    }
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
                    let $nod = unsafe { mem::transmute::<$r<_>, $r<Node4<$ty>>>(_b) };
                    $body
                },
                NODE_16 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<_>, $r<Node16<$ty>>>(_b) };
                    $body
                },
                NODE_48 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<_>, $r<Node48<$ty>>>(_b) };
                    $body
                },
                NODE_256 => {
                    #[allow(unused_unsafe)]
                    let $nod = unsafe { mem::transmute::<$r<_>, $r<Node256<$ty>>>(_b) };
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
    pub unsafe fn lookup_raw(&self, k: &T::Key) -> Option<*mut T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        unsafe fn lookup_raw_recursive<T: Element>(
            curr: &ChildPtr<T>,
            k: &T::Key,
            digits: &[u8],
        ) -> Option<*mut T> {
            // TODO: If Rust ever support proper tail-calls, this could be made tail-recursive.
            // In lieu of that, it's worth profiling this code to determine if an ugly iterative
            // rewrite would be worthwhile.
            // TODO: take consumed prefix into account?
            match curr.get_raw() {
                None => None,
                Some(Ok(leaf_node)) => {
                    if (*leaf_node).matches(k) {
                        Some(leaf_node)
                    } else {
                        None
                    }
                }
                Some(Err(inner_node)) => {
                    // handle prefixes now
                    (*inner_node)
                        .prefix_matches_optimistic(digits)
                        .and_then(|consumed| {
                            let new_digits = &digits[consumed..];
                            if new_digits.len() == 0 {
                                // Our digits were entirely consumed, but this is a non-leaf node.
                                // That means our node is not present.
                                return None;
                            }
                                with_node!(&*inner_node, nod, {
                                    nod.find_raw(new_digits[0]).and_then(|next_node| {
                                        lookup_raw_recursive(
                                            &*next_node,
                                            k,
                                            &new_digits[1..],
                                            )
                                    })
                                })
                        })
                }
            }
        }
        lookup_raw_recursive(&self.root, k, digits.as_slice())
    }
    pub unsafe fn delete_raw(&mut self, k: &T::Key) -> Option<T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        unsafe fn delete_raw_recursive<T: Element>(
            k: &T::Key,
            curr: &mut ChildPtr<T>,
            parent: Option<(u8, &mut ChildPtr<T>)>,
            digits: &[u8],
            // return the deleted node
        ) -> Option<T> {
            if curr.is_null() {
                return None;
            }
            unsafe fn move_val_out<T>(mut cptr: ChildPtr<T>) -> T {
                let res = {
                    let r = cptr.get_mut().unwrap().unwrap();
                    ptr::read(r)
                };
                mem::forget(cptr);
                res
            }

            let rest_opts = match curr.get_mut().unwrap() {
                Ok(leaf_node) => {
                    if digits.len() == 0 || leaf_node.matches(k) {
                        // we have a match! delete the leaf
                        if let Some((d, parent_ref)) = parent {
                            let (res, asgn) = with_node_mut!(
                                parent_ref.get_mut().unwrap().err().unwrap(),
                                node,
                                {
                                    match node.delete(d) {
                                        DeleteResult::Success(deleted) => {
                                            (Some(move_val_out(deleted)), None)
                                        }
                                        DeleteResult::Singleton { deleted, orphan } => {
                                            (Some(move_val_out(deleted)), Some(orphan))
                                        }
                                        DeleteResult::Failure => unreachable!(),
                                    }
                                }
                            );
                            if let Some(c_ptr) = asgn {
                                *parent_ref = c_ptr;
                            }
                            return res;
                        } else {
                            // This is the root, we'll set it to null below.
                            None
                        }
                    } else {
                        return None;
                    }
                }
                Err(inner_node) => {
                    let (matched, _) =
                        inner_node.get_matching_prefix(&digits[..], PhantomData as PhantomData<T>);
                    if matched == inner_node.count as usize {
                        // the prefix matched! we recur
                        debug_assert!(digits.len() > matched);
                        Some((inner_node as *mut RawNode<()>, matched))
                    } else {
                        // prefix was not a match, the key is not here
                        return None;
                    }
                    // if the prefix matches, recur, otherwise just bail out
                }
            };
            if let Some((inner_node, matched)) = rest_opts {
                let next_digit = digits[matched];
                with_node_mut!(&mut *inner_node, node, {
                    node.find_mut(next_digit).and_then(|c_ptr| {
                        // this wont compile
                        return delete_raw_recursive(
                            k,
                            c_ptr,
                            Some((next_digit, curr)),
                            &digits[matched + 1..],
                        );
                    })
                })
            } else {
                // we are in the root, set curr to null.
                let c_ptr = curr.swap_null();
                Some(move_val_out(c_ptr))
            }
        }
        let res = delete_raw_recursive(k, &mut self.root, None, &digits[..]);
        if res.is_some() {
            debug_assert!(self.len > 0);
            self.len -= 1;
        }
        res
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
                // Case 1: We found a null pointer, just replace it with a new leaf.
                let new_leaf = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                *curr = new_leaf;
                return Ok(());
            }
            // In order to get around the borrow checker, we need to move some inner variables out
            // of the case analysis and continue them with access to `curr`.
            //
            // This seems less error-prone than transmuting everything to raw pointers.
            enum Branch<T> {
                B1(SmallVec<[u8; 32]>, T),
                B2(*mut RawNode<Node4<T>>, u8),
            };
            let next_branch = match curr.get_mut().unwrap() {
                Ok(leaf_node) => {
                    // Case 2: We found a leaf node. We need to construct a new inner node with a the
                    // prefix corresponding to the shared prefix of this leaf node and `e`, add
                    // this leaf and `e` as a child to this new node, and replace the node as the
                    // root.
                    //
                    // Of course, we have already borrowed curr mutably, so we cannot accomplish
                    // these last few steps while we have still borrowed lead_node. We instead
                    // return the leaf's digits so we can do the rest of the loop outside of the
                    // match.
                    if leaf_node.matches(e.key()) {
                        // Found a matching leaf node. We swap in our value and return the old one.
                        leaf_node.replace_matching(&mut e);
                        return Err(e);
                    }
                    // found a leaf node, need to split it to a Node4 with two leaves
                    let mut leaf_digits = SmallVec::<[u8; 32]>::new();
                    leaf_digits.extend(leaf_node.key().digits());
                    debug_assert!(leaf_digits.len() <= digits.len());
                    Branch::B1(leaf_digits, e)
                }
                Err(inner_node) => {
                    // found an interior node. need to continue the search!
                    let (matched, min_ref) = inner_node
                        .get_matching_prefix(&digits[consumed..], PhantomData as PhantomData<T>);
                    if matched == digits[consumed..].len() {
                        // Case 3: we found an inner node, with a matching prefix.
                        //
                        // In this case we recursively insert our node into this inner node, making
                        // sure to update the 'consumed' variable appropriately.
                        consumed += matched;
                        // N.B what if consumed == digits.len()? the structure of the keys must
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
                            nod.count = cmp::min(nod.count, PREFIX_LEN as u32);
                            if let Some(next_ptr) = nod.find_mut(d) {
                                return insert_raw_recursive(next_ptr, e, digits, consumed + 1);
                            }
                            let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                            RawNode::insert(&mut (nod as *mut _), d, c_ptr);
                            return Ok(());
                        });
                    } else {
                        // Case 4: Our inner node shares a non-matching prefix with the current node.
                        //
                        // Here we have to figure out where the mismatch is and create a new parent
                        // node for the inner node and our current node.
                        unsafe fn adjust_prefix<R, T: Element>(
                            n: &mut RawNode<R>,
                            by: usize,
                            leaf: Option<*const T>,
                            consumed: usize,
                        ) {
                            let old_count = n.count as usize;
                            n.count -= by as u32;
                            let start: *const _ = &n.prefix[by];
                            // first we want to copy everything over
                            ptr::copy(start, &mut n.prefix[0], PREFIX_LEN - by);
                            if old_count > PREFIX_LEN {
                                let leaf_ref = &*leaf.unwrap();
                                for (p, d) in n.prefix[PREFIX_LEN - by..]
                                    .iter_mut()
                                    .zip(leaf_ref.key().digits().skip(consumed))
                                {
                                    *p = d;
                                }
                            }
                        }
                        let common_prefix_digits = &digits[consumed..matched];
                        let n4: Box<RawNode<Node4<T>>> =
                            make_node_with_prefix(&common_prefix_digits[..]);
                        debug_assert_eq!(n4.count as usize, common_prefix_digits.len());
                        consumed += n4.count as usize;
                        let by = inner_node.count as usize - common_prefix_digits.len();
                        adjust_prefix(inner_node, by, min_ref, consumed);
                        let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                        let mut n4_raw = Box::into_raw(n4);
                        // XXX: here we know that n4_raw will not get upgraded, that's the only
                        // reason re-using it here is safe.
                        RawNode::insert(&mut n4_raw, digits[consumed], c_ptr);
                        debug_assert!(inner_node.count > 0);

                        // We want to insert curr into n4, that means we need to figure out
                        // what its initial digit should be.
                        //
                        // We know that inner_node has some nonempty prefix (if it has an empty
                        // prefix we would have matched it). What we want is the first
                        // non-matching node in the prefix.
                        //
                        // Ideally we would do the rest of the operation! We cannot because of the
                        // same borrowing issue present with Case 2. We continue the operation on
                        // B2. Here's what we want to do:
                        //   let mut n4_cptr = ChildPtr::from_node(n4_raw);
                        //   mem::swap(curr, &mut n4_cptr);
                        //   RawNode::insert(&mut n4_raw, inner_node.prefix[0], n4_cptr);
                        //   return Ok(());
                        Branch::B2(n4_raw, inner_node.prefix[0])
                    }
                }
            };
            match next_branch {
                Branch::B1(leaf_digits, e) => {
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
                        leaf_digits[consumed + prefix_len - 1],
                        leaf_ptr,
                    );
                    RawNode::insert(&mut n4_raw, digits[consumed + prefix_len - 1], new_leaf);
                }
                Branch::B2(mut n4_raw, d) => {
                    let mut n4_cptr = ChildPtr::from_node(n4_raw);
                    mem::swap(curr, &mut n4_cptr);
                    RawNode::insert(&mut n4_raw, d, n4_cptr);
                }
            }
            Ok(())
        }
        let res = insert_raw_recursive(&mut self.root, elt, digits.as_slice(), 0);
        if res.is_ok() {
            self.len += 1;
        }
        res
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

    fn swap_null(&mut self) -> Self {
        let mut self_ptr = ChildPtr::null();
        mem::swap(self, &mut self_ptr);
        self_ptr
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

    unsafe fn get_raw(&self) -> Option<Result<*mut T, *mut RawNode<()>>> {
        if self.0 == 0 {
            None
        } else if self.0 & 1 == 1 {
            Some(Ok((self.0 & !1) as *mut T))
        } else {
            Some(Err(self.0 as *mut RawNode<()>))
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
#[derive(Debug)]
struct RawNode<Footer> {
    typ: NodeType,
    children: u16,
    count: u32,
    prefix: [u8; PREFIX_LEN],
    node: Footer,
}
impl RawNode<()> {
    fn get_matching_prefix<T: Element>(
        &self,
        digits: &[u8],
        _marker: PhantomData<T>,
    ) -> (usize, Option<*const T>) {
        let count = cmp::min(self.count as usize, PREFIX_LEN);
        for i in 0..count {
            if digits[i] != self.prefix[i] {
                return (i, None);
            }
        }
        if self.count as usize > PREFIX_LEN {
            let mut matches = PREFIX_LEN;
            with_node!(
                self,
                node,
                {
                    let min_node = node.get_min()
                        .expect("node with implicit prefix must be nonempty");
                    for (d, m) in digits[PREFIX_LEN..]
                        .iter()
                        .zip(min_node.key().digits().skip(PREFIX_LEN))
                    {
                        if *d != m {
                            break;
                        }
                        matches += 1;
                    }
                    (matches, Some(min_node as *const T))
                },
                T
            )
        } else {
            (count, None)
        }
    }
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
}

enum DeleteResult<T> {
    Failure,
    Success(ChildPtr<T>),
    Singleton {
        deleted: ChildPtr<T>,
        orphan: ChildPtr<T>,
    },
}

trait Node<T>: Sized {
    // insert assumes that 'd' is not present here already
    unsafe fn insert(curr: &mut *mut Self, d: u8, ptr: ChildPtr<T>);
    unsafe fn delete(&mut self, d: u8) -> DeleteResult<T>;
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

    impl<T> RawNode<Node4<T>> {
        fn find_internal(&self, d: u8) -> Option<(usize, *mut ChildPtr<T>)> {
            debug_assert!(self.children <= 4);
            for i in 0..4 {
                if self.node.keys[i] == d {
                    unsafe {
                        return Some((i, self.node.ptrs.get_unchecked(i) as *const _ as *mut _));
                    };
                }
                if i == self.children as usize {
                    return None;
                }
            }
            unreachable!()
        }
    }

    impl<T> Node<T> for RawNode<Node4<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            self.find_internal(d).map(|(_, ptr)| ptr)
        }

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            match self.find_internal(d) {
                None => DeleteResult::Failure,
                Some((ix, ptr)) => {
                    let mut deleted = ChildPtr::null();
                    mem::swap(&mut *ptr, &mut deleted);
                    if ix != 3 {
                        ptr::copy(&self.node.keys[ix + 1], &mut self.node.keys[ix], 4 - ix);
                    }
                    debug_assert!(self.children > 0);
                    self.children -= 1;
                    if self.children == 1 {
                        let mut c_ptr = ChildPtr::null();
                        mem::swap(&mut self.node.ptrs[self.node.keys[0] as usize], &mut c_ptr);
                        DeleteResult::Singleton {
                            deleted: deleted,
                            orphan: c_ptr,
                        }
                    } else {
                        DeleteResult::Success(deleted)
                    }
                }
            }
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

    impl<T> RawNode<Node16<T>> {
        fn find_internal(&self, d: u8) -> Option<(usize, *mut ChildPtr<T>)> {
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
                    let target = bits.trailing_zeros() as usize;
                    debug_assert!(target < 16);
                    Some((target, unsafe {
                        self.node.ptrs.get_unchecked(target) as *const _ as *mut _
                    }))
                };
            }
            #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"),
                          target_feature = "sse2")))]
            {
                // copy over Node4 implementation
                unimplemented!()
            }
        }
    }

    impl<T> Node<T> for RawNode<Node16<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            self.find_internal(d).map(|(_, ptr)| ptr)
        }

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            match self.find_internal(d) {
                None => DeleteResult::Failure,
                Some((ix, ptr)) => {
                    let mut deleted = ChildPtr::null();
                    mem::swap(&mut *ptr, &mut deleted);
                    if ix != 15 {
                        ptr::copy(&self.node.keys[ix + 1], &mut self.node.keys[ix], 16 - ix);
                    }
                    debug_assert!(self.children > 0);
                    self.children -= 1;
                    if self.children == 1 {
                        DeleteResult::Singleton {
                            deleted: deleted,
                            orphan: self.node.ptrs[self.node.keys[0] as usize].swap_null(),
                        }
                    } else {
                        DeleteResult::Success(deleted)
                    }
                }
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

    impl<T> RawNode<Node48<T>> {
        unsafe fn get_min_inner(&self) -> Option<(usize, *mut ChildPtr<T>)> {
            const KEYS_PER_WORD: usize = 8;
            const N_WORDS: usize = 256 / KEYS_PER_WORD;
            if self.children == 0 {
                return None;
            }
            let keys_words = mem::transmute::<&[u8; 256], &[u64; N_WORDS]>(&self.node.keys);
            for i in 0..N_WORDS {
                let word = keys_words[i];
                if word == 0 {
                    continue;
                }
                let word_bytes = mem::transmute::<u64, [u8; 8]>(word);
                for ii in 0..8 {
                    let b = word_bytes[ii];
                    if b != 0 {
                        let ix = b - 1;
                        return Some((i * KEYS_PER_WORD + ii, &self.node.ptrs[ix as usize] as *const _ as *mut _));
                    }
                }
            }
            unreachable!()
        }
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
            unsafe {
                self.get_min_inner()
                    .and_then(|(_, t)| match (*t).get().unwrap() {
                        Ok(t) => Some(t),
                        Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
                    })
            }
        }

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            match self.find_raw(d) {
                None => DeleteResult::Failure,
                Some(p) => {
                    // Found a pointer, swap out a null ChildPtr and set keys index to 0.
                    let deleted = (*p).swap_null();
                    self.node.keys[d as usize] = 0;
                    debug_assert!(self.children > 0);
                    self.children -= 1;
                    if self.children == 1 {
                        // TODO remove the first entry, it isn't required 
                        let (_, or_ptr) = self.get_min_inner().expect("Should be one more child");
                        DeleteResult::Singleton {
                            deleted: deleted,
                            orphan: (*or_ptr).swap_null(),
                        }
                    } else {
                        DeleteResult::Success(deleted)
                    }
                }
            }
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

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            if self.children == 0 || self.node.ptrs[d as usize].is_null() {
                return DeleteResult::Failure;
            }
            let deleted = self.node.ptrs[d as usize].swap_null();
            self.children -= 1;
            if self.children == 1 {
                for i in 0..256 {
                    let node = &mut self.node.ptrs[i];
                    if node.is_null() {
                        continue;
                    }
                    return DeleteResult::Singleton{
                        deleted: deleted,
                        orphan: node.swap_null(),
                    };
                }
            }
            DeleteResult::Success(deleted)
        }

        unsafe fn insert(curr: &mut *mut RawNode<Node256<T>>, d: u8, ptr: ChildPtr<T>) {
            debug_assert_eq!((**curr).typ, NODE_256);
            let slf: &mut RawNode<Node256<T>> = &mut **curr;
            debug_assert!(slf.children <= 256);
            debug_assert!(slf.node.ptrs[d as usize].is_null());
            slf.children += 1;
            ptr::write(slf.node.ptrs.get_unchecked_mut(d as usize), ptr);
        }
    }
}
