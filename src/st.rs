//! Single-threaded radix tree implementation based on HyPer's ART
extern crate fnv;
extern crate simd;
extern crate smallvec;

#[cfg(target_arch = "x86")]
use std::arch::x86::_mm_movemask_epi8;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_movemask_epi8;

use std::borrow::Borrow;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use self::node_variants::*;
use self::smallvec::{Array, SmallVec};
use super::Digital;
use super::byteorder::{ByteOrder, LittleEndian};

pub trait Element {
    type Key: for<'a> Digital<'a> + PartialOrd;
    fn key(&self) -> &Self::Key;
    fn matches(&self, k: &Self::Key) -> bool;
    fn replace_matching(&mut self, other: &mut Self);
}

pub struct ArtElement<T: for<'a> Digital<'a> + PartialOrd>(T);

impl<T: for<'a> Digital<'a> + PartialOrd> ArtElement<T> {
    pub fn new(t: T) -> ArtElement<T> {
        ArtElement(t)
    }
}

impl<T: for<'a> Digital<'a> + PartialOrd> Element for ArtElement<T> {
    type Key = T;
    fn key(&self) -> &T {
        &self.0
    }

    fn matches(&self, k: &Self::Key) -> bool {
        *k == self.0
    }

    fn replace_matching(&mut self, other: &mut ArtElement<T>) {
        debug_assert!(self.matches(other.key()));
        mem::swap(self, other);
    }
}

type RawMutRef<'a, T> = &'a mut RawNode<T>;
type RawRef<'a, T> = &'a RawNode<T>;

// TODO:
//  - Add pointers to hashtable to both toplevel and trait-level insert and delete methods
//  - Use these pointers to re-assign pointers upon growth and deletion
pub struct RawART<T: Element> {
    len: usize,
    root: ChildPtr<T>,
    prefix_target: usize,
    buckets: Buckets<T>,
}

struct Buckets<T>(Vec<(u64, MarkedPtr<T>)>);

impl<T> Buckets<T> {
    fn new(size: usize) -> Self {
        Buckets(
            (0..size.next_power_of_two())
                .map(|_| (0, MarkedPtr::null()))
                .collect::<Vec<_>>(),
        )
    }

    fn lookup(&self, bs: &[u8]) -> Option<MarkedPtr<T>> {
        use self::fnv::FnvHasher;
        use std::hash::Hasher;
        let mut hasher = FnvHasher::default();
        hasher.write(bs);
        debug_assert!(self.0.len().is_power_of_two());
        let key = hasher.finish() as usize & (self.0.len() - 1);
        let (i, ptr) = unsafe { self.0.get_unchecked(key) }.clone();
        if ptr.is_null() {
            None
        } else {
            let key = Self::read_u64(bs);
            if key == i {
                Some(ptr)
            } else {
                None
            }
        }
    }

    fn read_u64(bs: &[u8]) -> u64 {
        debug_assert!(bs.len() <= 8);
        let mut arr = [0 as u8; 8];
        unsafe { ptr::copy_nonoverlapping(&bs[0], &mut arr[0], cmp::min(bs.len(), 8)) };
        LittleEndian::read_u64(&arr[..])
    }

    fn insert(&mut self, bs: &[u8], ptr: MarkedPtr<T>) {
        use self::fnv::FnvHasher;
        use std::hash::Hasher;
        let mut hasher = FnvHasher::default();
        hasher.write(bs);
        debug_assert!(self.0.len().is_power_of_two());
        let h = hasher.finish() as usize & (self.0.len() - 1);
        let key = Self::read_u64(bs);
        unsafe { *self.0.get_unchecked_mut(h) = (key, ptr) };
    }
}

pub type ARTSet<T> = RawART<ArtElement<T>>;

impl<T: for<'a> Digital<'a> + PartialOrd> ARTSet<T> {
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: Borrow<T> + ?Sized,
    {
        unsafe { self.lookup_raw(key.borrow()).is_some() }
    }

    pub fn contains_val(&self, key: T) -> bool {
        self.contains(&key)
    }

    pub fn add(&mut self, k: T) -> bool {
        self.replace(k).is_some()
    }

    pub fn replace(&mut self, k: T) -> Option<T> {
        match unsafe { self.insert_raw(ArtElement::new(k)) } {
            Ok(()) => None,
            Err(ArtElement(t)) => Some(t),
        }
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<T>
    where
        Q: Borrow<T> + ?Sized,
    {
        unsafe { self.delete_raw(key.borrow()) }.map(|x| x.0)
    }

    pub fn remove_val(&mut self, key: T) -> bool {
        self.remove(&key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        Q: Borrow<T> + ?Sized,
    {
        self.take(key).is_some()
    }

    pub fn for_each_range<F: FnMut(&T)>(
        &self,
        mut f: F,
        lower_bound: Option<&T>,
        upper_bound: Option<&T>,
    ) {
        let mut lower_digits = SmallVec::<[u8; 16]>::new();
        let mut upper_digits = SmallVec::<[u8; 16]>::new();
        let mut ff = |x: &ArtElement<T>| f(&x.0);
        visit_leaf(
            &self.root,
            &mut ff,
            lower_bound.map(|x| {
                lower_digits.extend(x.digits());
                &lower_digits[..]
            }),
            upper_bound.map(|x| {
                upper_digits.extend(x.digits());
                &upper_digits[..]
            }),
            lower_bound,
            upper_bound,
        );
    }
}

macro_rules! with_node_inner {
    ($base_node: expr, $nod: ident, $body: expr, $r: tt) => {
        with_node_inner!($base_node, $nod, $body, $r, _)
    };
    ($base_node: expr, $nod: ident, $body: expr, $r: tt, $ty: tt) => {{
        let _b: $r<()> = $base_node;
        match _b.typ {
            NODE_4 => {
                #[allow(unused_unsafe)]
                let $nod = unsafe { mem::transmute::<$r<_>, $r<Node4<$ty>>>(_b) };
                $body
            }
            NODE_16 => {
                #[allow(unused_unsafe)]
                let $nod = unsafe { mem::transmute::<$r<_>, $r<Node16<$ty>>>(_b) };
                $body
            }
            NODE_48 => {
                #[allow(unused_unsafe)]
                let $nod = unsafe { mem::transmute::<$r<_>, $r<Node48<$ty>>>(_b) };
                $body
            }
            NODE_256 => {
                #[allow(unused_unsafe)]
                let $nod = unsafe { mem::transmute::<$r<_>, $r<Node256<$ty>>>(_b) };
                $body
            }
            _ => unreachable!(),
        }
    }};
}

macro_rules! with_node_mut {
    ($base_node: expr, $nod: ident, $body: expr) => {
        with_node_mut!($base_node, $nod, $body, _)
    };
    ($base_node: expr, $nod: ident, $body: expr, $ty: tt) => {
        with_node_inner!($base_node, $nod, $body, RawMutRef, $ty)
    };
}

macro_rules! with_node {
    ($base_node: expr, $nod: ident, $body: expr) => {
        with_node!($base_node, $nod, $body, _)
    };
    ($base_node: expr, $nod: ident, $body: expr, $ty: tt) => {
        with_node_inner!($base_node, $nod, $body, RawRef, $ty)
    };
}
enum InsertResult<T> {
    Failure(T),
    Replaced(T),
    Success,
}

// TODO:
// 1 Use full predicate to avoid unnecessary heap allocation [check]
// 2 For insertions that create an interior node (case 3 and case 4),
//   need to insert node into hash table. This can be done by passing a
//   reference to the function. [check]
// 3 For removals, any place that removes an interior node (basically
//   the case when an orphan is returned) check the old nodes `consumed`
//   value and remove it from the table if necessary. [half-done]
// 4 Clean this shit up!

impl<T: Element> RawART<T> {
    pub fn new() -> Self {
        RawART::with_prefix_buckets(4, 8192)
    }

    pub fn with_prefix_buckets(prefix_len: usize, buckets: usize) -> Self {
        assert!(prefix_len <= 8);
        RawART {
            len: 0,
            root: ChildPtr::null(),
            buckets: Buckets::new(buckets),
            prefix_target: prefix_len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn hash_lookup(&self, digits: &[u8]) -> Option<MarkedPtr<T>> {
        if digits.len() < self.prefix_target {
            None
        } else {
            self.buckets.lookup(&digits[0..self.prefix_target])
        }
    }

    // replace with NonNull
    pub unsafe fn lookup_raw(&self, k: &T::Key) -> Option<*mut T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        unsafe fn lookup_raw_recursive<T: Element>(
            curr: MarkedPtr<T>,
            k: &T::Key,
            digits: &[u8],
            dont_check: bool,
        ) -> Option<*mut T> {
            match curr.get_raw() {
                None => None,
                Some(Ok(leaf_node)) => {
                    if (dont_check && digits.len() == 0) || (*leaf_node).matches(k) {
                        Some(leaf_node)
                    } else {
                        None
                    }
                }
                Some(Err(inner_node)) => {
                    // handle prefixes now
                    (*inner_node).prefix_matches_optimistic(digits).and_then(
                        |(dont_check_new, consumed)| {
                            let new_digits = &digits[consumed..];
                            if new_digits.len() == 0 {
                                // Our digits were entirely consumed, but this is a non-leaf node.
                                // That means our node is not present.
                                return None;
                            }
                            with_node!(&*inner_node, nod, {
                                nod.find_raw(new_digits[0]).and_then(|next_node| {
                                    lookup_raw_recursive(
                                        (&*next_node).to_marked(),
                                        k,
                                        &new_digits[1..],
                                        dont_check && dont_check_new,
                                    )
                                })
                            })
                        },
                    )
                }
            }
        }

        let (node_ref, slice) = if let Some(ptr) = self.hash_lookup(digits.as_slice()) {
            (ptr, &digits[self.prefix_target..])
        } else {
            (self.root.to_marked(), digits.as_slice())
        };
        lookup_raw_recursive(node_ref, k, slice, true)
    }

    pub unsafe fn delete_raw(&mut self, k: &T::Key) -> Option<T> {
        // XXX There's a rare bug here:
        // "
        // thread 'st::tests::string_set_behavior' panicked at 'Deletion failed immediately for ["en\u{16}P{" : [102, 111, 23, 81, 124, 0]]', src/st.rs:1888:13
        // "

        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        unsafe fn delete_raw_recursive<T: Element>(
            k: &T::Key,
            curr: &mut ChildPtr<T>,
            parent: Option<(u8, &mut ChildPtr<T>)>,
            digits: &[u8],
            mut consumed: usize,
            target: usize,
            buckets: &mut Buckets<T>,
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
                    /* digits.len() == 0 || */
                    if leaf_node.matches(k) {
                        // we have a match! delete the leaf
                        if let Some((d, parent_ref)) = parent {
                            let (res, asgn) = with_node_mut!(
                                parent_ref.get_mut().unwrap().err().unwrap(),
                                node,
                                {
                                    match node.delete(d) {
                                        DeleteResult::Success(deleted) => {
                                            // TODO: free up space here
                                            (Some(move_val_out(deleted)), None)
                                        }
                                        DeleteResult::Singleton { deleted, orphan } => {
                                            (Some(move_val_out(deleted)), Some(orphan))
                                        }
                                        DeleteResult::Failure => unreachable!(),
                                    }
                                }
                            );
                            if let Some(mut c_ptr) = asgn {
                                let mut switch = false;
                                if let Err(inner) = c_ptr.get_mut().unwrap() {
                                    inner.append_prefix(d);
                                    if inner.consumed as usize == target {
                                        switch = true;
                                    }
                                }
                                // need to add d as a prefix to c_ptr
                                *parent_ref = c_ptr;
                                if switch {
                                    buckets.insert(&digits[0..consumed], parent_ref.to_marked());
                                    debug_assert!(parent_ref.get().unwrap().is_err());
                                }
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
                    debug_assert_eq!(consumed, inner_node.consumed as usize);
                    let (matched, _) = inner_node.get_matching_prefix(
                        digits,
                        consumed,
                        PhantomData as PhantomData<T>,
                    );
                    // if the prefix matches, recur, otherwise just bail out
                    if matched == inner_node.count as usize {
                        // the prefix matched! we recur below
                        debug_assert!(digits.len() > matched);
                        Some((inner_node as *mut RawNode<()>, matched))
                    } else {
                        // prefix was not a match, the key is not here
                        return None;
                    }
                }
            };
            if let Some((inner_node, matched)) = rest_opts {
                let next_digit = digits[consumed + matched];
                with_node_mut!(&mut *inner_node, node, {
                    node.find_mut(next_digit).and_then(|c_ptr| {
                        consumed += matched + 1;
                        delete_raw_recursive(
                            k,
                            c_ptr,
                            Some((next_digit, curr)),
                            digits,
                            consumed,
                            target,
                            buckets,
                        )
                    })
                })
            } else {
                // we are in the root, set curr to null.
                let c_ptr = curr.swap_null();
                Some(move_val_out(c_ptr))
            }
        }
        let res = delete_raw_recursive(
            k,
            &mut self.root,
            None,
            &digits[..],
            0,
            self.prefix_target,
            &mut self.buckets,
        );
        if res.is_some() {
            debug_assert!(self.len > 0);
            self.len -= 1;
        }
        res
    }

    pub unsafe fn insert_raw(&mut self, elt: T) -> Result<(), T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(elt.key().digits());

        unsafe fn insert_raw_recursive<T: Element>(
            curr: MarkedPtr<T>,
            mut e: T,
            digits: &[u8],
            mut consumed: usize,
            pptr: Option<*mut ChildPtr<T>>,
            buckets: &mut Buckets<T>,
            target: usize,
        ) -> InsertResult<T> {
            use self::InsertResult::*;
            debug_assert!(consumed <= digits.len());
            if curr.is_null() {
                // Case 1: We found a null pointer, just replace it with a new leaf.
                let new_leaf = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                (*pptr.unwrap()) = new_leaf;
                return Success;
            }
            match curr.get_raw().unwrap() {
                Ok(ln) => {
                    debug_assert!(pptr.is_some());
                    // Case 2: We found a leaf node. We need to construct a new inner node with a the
                    // prefix corresponding to the shared prefix of this leaf node and `e`, add
                    // this leaf and `e` as a child to this new node, and replace the node as the
                    // root.
                    //
                    // Of course, we have already borrowed curr mutably, so we cannot accomplish
                    // these last few steps while we have still borrowed lead_node. We instead
                    // return the leaf's digits so we can do the rest of the loop outside of the
                    // match.
                    let leaf_node = &mut *ln;
                    if leaf_node.matches(e.key()) {
                        // Found a matching leaf node. We swap in our value and return the old one.
                        leaf_node.replace_matching(&mut e);
                        return Replaced(e);
                    }
                    // found a leaf node, need to split it to a Node4 with two leaves
                    let mut leaf_digits = SmallVec::<[u8; 32]>::new();
                    leaf_digits.extend(leaf_node.key().digits());
                    // Branch::B1(leaf_digits, e)
                    let pp = pptr.unwrap();
                    let n4: Box<RawNode<Node4<T>>> = make_node_from_common_prefix(
                        &leaf_digits.as_slice()[consumed..],
                        &digits[consumed..],
                        consumed as u32,
                    );
                    let prefix_len = n4.count as usize;
                    let mut n4_raw = Box::into_raw(n4);
                    let mut leaf_ptr = ChildPtr::from_node(n4_raw);
                    let new_leaf = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                    mem::swap(&mut *pp, &mut leaf_ptr);
                    if consumed <= target && target < consumed + (*n4_raw).count as usize {
                        buckets.insert(&digits[0..consumed], (*pp).to_marked());
                        debug_assert!((*pp).get().unwrap().is_err());
                    }
                    // n4_raw has now replaced the leaf, we need to reinsert the leaf, along with
                    // our child pointer.
                    debug_assert!(consumed + prefix_len < leaf_digits.len(),
                                  "leaf digits ({:?}) out of space due to prefix shared with d={:?} (consumed={:?})",
                                  &leaf_digits[..],
                                  digits,
                                  consumed);
                    (*n4_raw)
                        .insert(leaf_digits[consumed + prefix_len], leaf_ptr, None)
                        .unwrap();
                    (*n4_raw)
                        .insert(digits[consumed + prefix_len], new_leaf, None)
                        .unwrap()
                }
                Err(inn) => {
                    let inner_node = &mut *inn;
                    debug_assert_eq!(consumed, inner_node.consumed as usize);
                    // found an interior node. need to continue the search!
                    let (matched, min_ref) = inner_node.get_matching_prefix(
                        &digits[..],
                        consumed,
                        PhantomData as PhantomData<T>,
                    );

                    if matched == inner_node.count as usize {
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
                                let pp = Some(next_ptr as *mut _);
                                return insert_raw_recursive(
                                    next_ptr.to_marked(),
                                    e,
                                    digits,
                                    consumed + 1,
                                    pp,
                                    buckets,
                                    target,
                                );
                            }
                            if nod.is_full() && pptr.is_none() {
                                return Failure(e);
                            }
                            let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                            let _r = nod.insert(d, c_ptr, pptr);
                            debug_assert!(_r.is_ok());
                            return Success;
                        });
                    } else {
                        let inner_d = inner_node.prefix[matched];
                        if pptr.is_none() {
                            return Failure(e);
                        }
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
                            debug_assert!(by > 0);
                            debug_assert!(
                                by <= n.count as usize,
                                "by={:?} > n.count={:?}",
                                by,
                                n.count
                            );
                            let old_count = n.count as usize;
                            n.count -= by as u32;
                            let start: *const _ = &n.prefix[by];
                            ptr::copy(start, &mut n.prefix[0], n.count as usize);
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
                        debug_assert!(
                            inner_node.count > 0,
                            "Found 0 inner_node.count in split case, matched={:?}",
                            matched
                        );
                        let common_prefix_digits = &digits[consumed..consumed + matched];
                        debug_assert_eq!(common_prefix_digits.len(), matched);
                        let n4: Box<RawNode<Node4<T>>> =
                            make_node_with_prefix(&common_prefix_digits[..], consumed as u32);
                        inner_node.consumed += n4.count + 1;
                        debug_assert_eq!(n4.count as usize, common_prefix_digits.len());
                        consumed += n4.count as usize;
                        let by = matched + 1;
                        adjust_prefix(inner_node, by, min_ref, consumed);
                        let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                        let mut n4_raw = Box::into_raw(n4);
                        let _r = (*n4_raw).insert(digits[consumed], c_ptr, None);
                        debug_assert!(_r.is_ok());
                        let pp = pptr.unwrap();
                        let mut n4_cptr = ChildPtr::from_node(n4_raw);
                        mem::swap(&mut *pp, &mut n4_cptr);
                        if (*n4_raw).consumed as usize == target {
                            buckets.insert(&digits[0..target], (*pp).to_marked());
                            debug_assert!((*pp).get().unwrap().is_err());
                        }
                        (*n4_raw).insert(inner_d, n4_cptr, None).unwrap()
                    }
                }
            };
            Success
        }
        let e = {
            let (node_ref, consumed, pptr) = if let Some(ptr) = self.hash_lookup(digits.as_slice())
            {
                (ptr, self.prefix_target, None)
            } else {
                let root_alias = Some(&mut self.root as *mut _);
                (self.root.to_marked(), 0, root_alias)
            };
            match insert_raw_recursive(
                node_ref,
                elt,
                digits.as_slice(),
                consumed,
                pptr,
                &mut self.buckets,
                self.prefix_target,
            ) {
                InsertResult::Failure(e) => e,
                InsertResult::Success => {
                    self.len += 1;
                    return Ok(());
                }
                InsertResult::Replaced(t) => {
                    return Err(t);
                }
            }
        };
        // Hash-indexed inserts can fail, because we don't have access to a parent pointer.
        let root_alias = Some(&mut self.root as *mut _);
        match insert_raw_recursive(
            self.root.to_marked(),
            e,
            digits.as_slice(),
            0,
            root_alias,
            &mut self.buckets,
            self.prefix_target,
        ) {
            InsertResult::Success => {
                self.len += 1;
                Ok(())
            }
            InsertResult::Replaced(t) => Err(t),
            InsertResult::Failure(_) => unreachable!(),
        }
    }
}

// may want to get the algorithmics right first. Just assume you're storing usize and have
// Either<Leaf, Option<Box<Node...>>> as the child pointers.
// TODO: use NonNull everywhere

struct MarkedPtr<T>(usize, PhantomData<T>);
impl<T> Clone for MarkedPtr<T> {
    fn clone(&self) -> Self {
        MarkedPtr(self.0, PhantomData)
    }
}
struct ChildPtr<T>(MarkedPtr<T>);

impl<T> ::std::ops::Deref for ChildPtr<T> {
    type Target = MarkedPtr<T>;
    fn deref(&self) -> &MarkedPtr<T> {
        &self.0
    }
}

impl<T> ::std::ops::DerefMut for ChildPtr<T> {
    fn deref_mut(&mut self) -> &mut MarkedPtr<T> {
        &mut self.0
    }
}

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

impl<T> ::std::fmt::Debug for MarkedPtr<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "MarkedPtr({:?})", self.0)
    }
}

impl<T> ::std::fmt::Debug for ChildPtr<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "ChildPtr({:?})", (self.0).0)
    }
}

impl<T> ChildPtr<T> {
    fn null() -> Self {
        ChildPtr(MarkedPtr::null())
    }

    fn from_node<R>(p: *mut RawNode<R>) -> Self {
        ChildPtr(MarkedPtr::from_node(p))
    }

    fn from_leaf(p: *mut T) -> Self {
        ChildPtr(MarkedPtr::from_leaf(p))
    }

    fn swap_null(&mut self) -> Self {
        let mut self_ptr = ChildPtr::null();
        mem::swap(self, &mut self_ptr);
        self_ptr
    }

    unsafe fn to_marked(&self) -> MarkedPtr<T> {
        ptr::read(&self.0)
    }
}

impl<T> MarkedPtr<T> {
    fn null() -> Self {
        MarkedPtr(0, PhantomData)
    }

    fn from_node<R>(p: *mut RawNode<R>) -> Self {
        debug_assert!(!p.is_null());
        MarkedPtr(p as usize, PhantomData)
    }

    fn from_leaf(p: *mut T) -> Self {
        debug_assert!(!p.is_null());
        MarkedPtr((p as usize) | 1, PhantomData)
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
    consumed: u32,
    prefix: [u8; PREFIX_LEN],
    node: Footer,
}

impl<T> RawNode<T> {
    fn append_prefix(&mut self, d: u8) {
        unsafe {
            ptr::copy(&self.prefix[0], &mut self.prefix[1], PREFIX_LEN - 1);
        }
        self.prefix[0] = d;
        self.count += 1;
        self.consumed -= 1;
    }
}

impl RawNode<()> {
    fn get_matching_prefix<T: Element>(
        &self,
        digits: &[u8],
        consumed: usize,
        _marker: PhantomData<T>,
    ) -> (usize, Option<*const T>) {
        let count = cmp::min(self.count as usize, PREFIX_LEN);
        for i in 0..count {
            if digits[consumed + i] != self.prefix[i] {
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
                    for (d, m) in digits[consumed + PREFIX_LEN..]
                        .iter()
                        .zip(min_node.key().digits().skip(consumed + PREFIX_LEN))
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
    fn prefix_matches_optimistic(&self, digits: &[u8]) -> Option<(bool, usize)> {
        let count = self.count as usize;
        if digits.len() < count {
            return None;
        }
        for i in 0..cmp::min(count, PREFIX_LEN) {
            if digits[i] != self.prefix[i] {
                return None;
            }
        }
        Some((count <= PREFIX_LEN, count))
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

trait Node<T: Element>: Sized {
    // insert assumes that 'd' is not present in the node. This is enforced in debug buids
    /// TODO: pptr becomes an option, insert returns Result
    ///       IRR can take two pointers as well?
    ///       using this pattern helps us avoid missing things areas
    ///
    ///       insert returns a Result<(),ChildPtr<T>>
    unsafe fn insert(
        &mut self,
        d: u8,
        ptr: ChildPtr<T>,
        // Error == ptr, indicates there was no space _and_ could not upgrade
        pptr: Option<*mut ChildPtr<T>>,
    ) -> Result<(), ChildPtr<T>>;
    unsafe fn delete(&mut self, d: u8) -> DeleteResult<T>;
    fn is_full(&self) -> bool;
    fn get_min(&self) -> Option<&T>;
    fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>>;
    fn find(&self, d: u8) -> Option<&ChildPtr<T>> {
        self.find_raw(d).map(|raw_ptr| unsafe { &*raw_ptr })
    }
    fn find_mut(&self, d: u8) -> Option<&mut ChildPtr<T>> {
        self.find_raw(d).map(|raw_ptr| unsafe { &mut *raw_ptr })
    }

    fn for_each<F: FnMut(&T)>(
        &self,
        f: &mut F,
        lower: Option<&[u8]>,
        upper: Option<&[u8]>,
        lval: Option<&T::Key>,
        rval: Option<&T::Key>,
    );
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

fn make_node_with_prefix<T>(prefix: &[u8], consumed: u32) -> Box<RawNode<Node4<T>>> {
    let mut new_node = Box::new(RawNode {
        typ: NODE_4,
        children: 0,
        consumed: consumed,
        count: prefix.len() as u32,
        prefix: [0; PREFIX_LEN],
        node: Node4 {
            keys: [0; 4],
            ptrs: unsafe { mem::transmute::<[usize; 4], [ChildPtr<T>; 4]>([0 as usize; 4]) },
        },
    });
    let new_len = cmp::min(prefix.len(), PREFIX_LEN);
    if prefix.len() > 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                &prefix[0] as *const _,
                &mut new_node.prefix[0] as *mut _,
                new_len,
            );
        }
    }
    new_node
}

fn make_node_from_common_prefix<T>(d1: &[u8], d2: &[u8], consumed: u32) -> Box<RawNode<Node4<T>>> {
    let mut common_prefix_digits = SmallVec::<[u8; 32]>::new();
    get_matching_prefix_slice(d1.iter(), d2.iter(), &mut common_prefix_digits);
    make_node_with_prefix(&common_prefix_digits[..], consumed)
}

struct Node4<T> {
    keys: [u8; 4],
    ptrs: [ChildPtr<T>; 4],
}

impl<T> ::std::fmt::Debug for Node4<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "Node4({:?}, {:?})", self.keys, &self.ptrs[..])
    }
}

fn visit_leaf<T, F>(
    c: &ChildPtr<T>,
    f: &mut F,
    mut lower: Option<&[u8]>,
    mut upper: Option<&[u8]>,
    lval: Option<&T::Key>,
    rval: Option<&T::Key>,
) where
    F: FnMut(&T),
    T: Element,
{
    fn advance_by(s: &mut Option<&[u8]>, by: usize) {
        if s.is_none() {
            return;
        }
        debug_assert!(s.unwrap().len() > 0);
        let slice = s.unwrap();
        if slice.len() <= by {
            *s = None;
        }
        *s = Some(&slice[by..]);
    }

    /// An over the prefix of a `RawNode`. This is used to encapsulate the tricky "implicit prefix"
    /// semantics which are occasionally required for large keys.
    struct PrefixIter<'a, T: Element + 'a> {
        ix: usize,
        len: usize,
        node: &'a RawNode<()>,
        _min: SmallVec<[u8; 2]>,
        _marker: PhantomData<T>,
    }

    impl<'a, T: Element + 'a> PrefixIter<'a, T> {
        fn new(node: &'a RawNode<()>) -> Self {
            PrefixIter {
                ix: 0,
                len: node.count as usize,
                node: node,
                _min: SmallVec::new(),
                _marker: PhantomData,
            }
        }

        fn reset(&mut self) {
            self.ix = 0;
        }
    }

    impl<'a, T: 'a + Element> Iterator for PrefixIter<'a, T> {
        type Item = u8;

        fn next(&mut self) -> Option<u8> {
            if self.ix >= self.len {
                return None;
            }
            if self.ix < PREFIX_LEN {
                let res = self.node.prefix[self.ix];
                self.ix += 1;
                return Some(res);
            }
            None
            // TODO: for now, we don't do the implicit prefix and rely on the check at the end to
            // filter out any false positives.
            //
            // if self.ix == PREFIX_LEN && self.min.len() == 0 {
            //     self.min.extend(
            //         with_node!(self.node, node, node.get_min(), T)
            //             .unwrap()
            //             .key()
            //             .digits(),
            //     );
            // }

            // //XXX: this is wrong, isn't it? Need to find out _where_ it is in min.
            // //Which means we have to pass consumed through for_each. Oh well.
            // let res = self.min[self.ix];
            // self.ix += 1;
            // Some(res)
        }
    }
    match unsafe { c.get() } {
        None => {}
        Some(Ok(ref leaf)) => {
            if let Some(up) = rval {
                if up <= leaf.key() {
                    return;
                }
            }
            // N.B: If we choose to fully handle implicit prefixes then this check should be
            // unnecessary.
            if let Some(low) = lval {
                if low > leaf.key() {
                    return;
                }
            }
            f(leaf)
        }
        Some(Err(inner)) => {
            let mut iter = PrefixIter::<T>::new(inner);
            if let Some(slice) = lower {
                for (l, byte) in slice.iter().zip(&mut iter) {
                    if byte < *l {
                        return;
                    }
                }
            }
            iter.reset();
            if let Some(slice) = upper {
                for (h, byte) in slice.iter().zip(&mut iter) {
                    if byte > *h {
                        return;
                    }
                }
            }
            advance_by(&mut lower, inner.count as usize);
            advance_by(&mut upper, inner.count as usize);
            with_node!(inner, node, { node.for_each(f, lower, upper, lval, rval) })
        }
    }
}

mod node_variants {
    use super::*;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NodeType(u16);
    pub const NODE_4: NodeType = NodeType(1);
    pub const NODE_16: NodeType = NodeType(2);
    pub const NODE_48: NodeType = NodeType(3);
    pub const NODE_256: NodeType = NodeType(4);

    fn advance_or(s: &mut Option<&[u8]>, b: usize) -> usize {
        if s.is_none() {
            return b;
        }
        debug_assert!(s.unwrap().len() > 0);
        let slice = s.unwrap();
        let res = slice[0] as usize;
        if slice.len() == 1 {
            *s = None;
        } else {
            *s = Some(&slice[1..]);
        }
        res
    }

    // (very) ad-hoc polymorphism!
    macro_rules! n416_delete {
        ($slf: expr, $d: expr) => {
            match $slf.find_internal($d) {
                None => DeleteResult::Failure,
                Some((ix, ptr)) => {
                    let deleted = (*ptr).swap_null();
                    if ix + 1 < $slf.node.keys[..].len() {
                        ptr::copy(
                            &$slf.node.keys[ix + 1],
                            &mut $slf.node.keys[ix],
                            $slf.children as usize - ix,
                        );
                        ptr::copy(
                            &$slf.node.ptrs[ix + 1],
                            &mut $slf.node.ptrs[ix],
                            $slf.children as usize - ix,
                        );
                    }
                    debug_assert!($slf.children > 0);
                    $slf.children -= 1;
                    ptr::write(
                        &mut $slf.node.ptrs[$slf.children as usize],
                        ChildPtr::null(),
                    );
                    if $slf.children == 1 {
                        let mut c_ptr = ChildPtr::null();
                        debug_assert!(!$slf.node.ptrs[0].is_null());
                        mem::swap(&mut $slf.node.ptrs[0], &mut c_ptr);
                        DeleteResult::Singleton {
                            deleted: deleted,
                            orphan: c_ptr,
                        }
                    } else {
                        DeleteResult::Success(deleted)
                    }
                }
            }
        };
    }

    fn is_sorted(slice: &[u8]) -> bool {
        let mut v: Vec<u8> = Vec::new();
        v.extend(slice);
        v.sort();
        let res = &v[..] == slice;
        if !res {
            eprintln!("Not sorted! {:?} != {:?}", slice, v);
        }
        res
    }

    macro_rules! n416_foreach {
        ($slf: expr, $f: expr, $lower: expr, $upper: expr, $lval: expr, $uval: expr) => {{
            debug_assert!(is_sorted(&$slf.node.keys[..$slf.children as usize]));
            let low = advance_or(&mut $lower, 0);
            let high = advance_or(&mut $upper, 255);
            let children = $slf.children as usize;
            for i in 0..children {
                let k = $slf.node.keys[i] as usize;
                if k < low {
                    continue;
                }
                if k > high {
                    break;
                }
                let low = if k == low { $lower } else { None };
                let high = if k == high { $upper } else { None };
                visit_leaf(&$slf.node.ptrs[i], $f, low, high, $lval, $uval)
            }
        }};
    }

    impl<T> RawNode<Node4<T>> {
        fn find_internal(&self, d: u8) -> Option<(usize, *mut ChildPtr<T>)> {
            debug_assert!(self.children <= 4);
            for i in 0..4 {
                if i == self.children as usize {
                    break;
                }
                if self.node.keys[i] == d {
                    unsafe {
                        return Some((i, self.node.ptrs.get_unchecked(i) as *const _ as *mut _));
                    };
                }
            }
            None
        }
    }

    impl<T: Element> Node<T> for RawNode<Node4<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            self.find_internal(d).map(|(_, ptr)| ptr)
        }

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            n416_delete!(self, d)
        }

        fn get_min(&self) -> Option<&T> {
            debug_assert!(self.children <= 4);
            if self.children == 0 {
                return None;
            }
            // we keep the child list sorted, so we recur at '0'
            match unsafe { self.node.ptrs[0 as usize].get().unwrap() } {
                Ok(t) => Some(t),
                Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
            }
        }

        fn is_full(&self) -> bool {
            self.children == 4
        }

        unsafe fn insert(
            &mut self,
            d: u8,
            ptr: ChildPtr<T>,
            pptr: Option<*mut ChildPtr<T>>,
        ) -> Result<(), ChildPtr<T>> {
            debug_assert!(self.find_raw(d).is_none());
            if self.children == 4 {
                if let Some(pp) = pptr {
                    let new_node = &mut *Box::into_raw(Box::new(RawNode {
                        typ: NODE_16,
                        children: self.children,
                        consumed: self.consumed,
                        count: self.count,
                        prefix: self.prefix,
                        node: Node16 {
                            keys: [0; 16],
                            ptrs: mem::transmute::<[usize; 16], [ChildPtr<T>; 16]>(
                                [0 as usize; 16],
                            ),
                        },
                    }));
                    ptr::swap_nonoverlapping(&mut self.node.keys[0], &mut new_node.node.keys[0], 4);
                    ptr::swap_nonoverlapping(&mut self.node.ptrs[0], &mut new_node.node.ptrs[0], 4);
                    let new_cptr = ChildPtr::from_node(new_node);
                    *pp = new_cptr;
                    let res = new_node.insert(d, ptr, None);
                    debug_assert!(res.is_ok());
                    return res;
                } else {
                    return Err(ptr);
                }
            }
            for i in 0..4 {
                if i == (self.children as usize) {
                    // found an empty slot!
                    debug_assert!(self.node.ptrs[i].is_null());
                    self.node.keys[i] = d;
                    self.node.ptrs[i] = ptr;
                    self.children += 1;
                    debug_assert!(is_sorted(&self.node.keys[..self.children as usize]));
                    return Ok(());
                }
                let cur_digit = self.node.keys[i];
                debug_assert!(
                    cur_digit != d,
                    "Found matching current digit! cur_digit={:?}, prefix={:?}, count={:?}, children={:?} keys={:?}",
                    cur_digit,
                    &self.prefix[..],
                    self.count,
                    self.children,
                    &self.node.keys[..],
                );
                if cur_digit > d {
                    // we keep the list sorted, so we need to move all other entries ahead by 1
                    // slot. This is not safe if it's already full.
                    place_in_hole_at(&mut self.node.keys[..], i, d, 4);
                    place_in_hole_at(&mut self.node.ptrs[..], i, ptr, 4);
                    self.children += 1;
                    debug_assert!(is_sorted(&self.node.keys[..self.children as usize]));
                    return Ok(());
                }
            }
            unreachable!()
        }

        fn for_each<F: FnMut(&T)>(
            &self,
            f: &mut F,
            mut lower: Option<&[u8]>,
            mut upper: Option<&[u8]>,
            lval: Option<&T::Key>,
            rval: Option<&T::Key>,
        ) {
            n416_foreach!(self, f, lower, upper, lval, rval)
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
                let bits = unsafe { _mm_movemask_epi8(mem::transmute(comps)) } & mask;
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

    impl<T: Element> Node<T> for RawNode<Node16<T>> {
        fn is_full(&self) -> bool {
            self.children == 16
        }
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            self.find_internal(d).map(|(_, ptr)| ptr)
        }

        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            n416_delete!(self, d)
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

        unsafe fn insert(
            &mut self,
            d: u8,
            ptr: ChildPtr<T>,
            pptr: Option<*mut ChildPtr<T>>,
        ) -> Result<(), ChildPtr<T>> {
            debug_assert!(self.find_raw(d).is_none());
            let mask = (1 << (self.children as usize)) - 1;
            if self.children == 16 {
                if let Some(pp) = pptr {
                    // upgrade
                    let new_node = &mut *Box::into_raw(Box::new(RawNode {
                        typ: NODE_48,
                        children: 16,
                        count: self.count,
                        consumed: self.consumed,
                        prefix: self.prefix,
                        node: Node48 {
                            keys: [0; 256],
                            ptrs: mem::transmute::<[usize; 48], [ChildPtr<T>; 48]>(
                                [0 as usize; 48],
                            ),
                        },
                    }));
                    for i in 0..16 {
                        let ix = self.node.keys[i] as usize;
                        mem::swap(
                            self.node.ptrs.get_unchecked_mut(i),
                            new_node.node.ptrs.get_unchecked_mut(i),
                        );
                        new_node.node.keys[ix] = i as u8 + 1;
                    }
                    let new_cptr = ChildPtr::from_node(new_node);
                    *pp = new_cptr;
                    let res = new_node.insert(d, ptr, None);
                    debug_assert!(res.is_ok());
                    return Ok(());
                } else {
                    return Err(ptr);
                }
            }
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "sse2"))]
            {
                let ks = simd::u8x16::load(&self.node.keys[..], 0);
                let d_splat = simd::u8x16::splat(d);
                let comps = d_splat.lt(ks);
                let bits: i32 = _mm_movemask_epi8(mem::transmute(comps)) & mask;
                let zeros = bits.trailing_zeros();
                let target = if zeros == 32 {
                    self.children as usize
                } else {
                    zeros as usize
                };
                place_in_hole_at(&mut self.node.keys[..], target, d, 16);
                place_in_hole_at(&mut self.node.ptrs[..], target, ptr, 16);
            }
            #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"),
                          target_feature = "sse2")))]
            {
                // copy over Node16 implementation
                unimplemented!()
            }
            self.children += 1;
            debug_assert!(is_sorted(&self.node.keys[..self.children as usize]));
            return Ok(());
        }

        fn for_each<F: FnMut(&T)>(
            &self,
            f: &mut F,
            mut lower: Option<&[u8]>,
            mut upper: Option<&[u8]>,
            lval: Option<&T::Key>,
            rval: Option<&T::Key>,
        ) {
            n416_foreach!(self, f, lower, upper, lval, rval)
        }
    }

    pub struct Node48<T> {
        keys: [u8; 256],
        ptrs: [ChildPtr<T>; 48],
    }

    impl<T> RawNode<Node48<T>> {
        unsafe fn get_min_inner(&self) -> Option<(usize, *mut ChildPtr<T>)> {
            for i in 0..256 {
                let d = self.node.keys[i];
                if d == 0 {
                    continue;
                }
                let ix = d as usize - 1;
                return Some((ix, &self.node.ptrs[ix] as *const _ as *mut _));
            }
            None
            // potentially optimized solution below:
            // const KEYS_PER_WORD: usize = 8;
            // const N_WORDS: usize = 256 / KEYS_PER_WORD;
            // if self.children == 0 {
            //     return None;
            // }
            // let keys_words = mem::transmute::<&[u8; 256], &[u64; N_WORDS]>(&self.node.keys);
            // for i in 0..N_WORDS {
            //     let word = keys_words[i];
            //     if word == 0 {
            //         continue;
            //     }
            //     let word_bytes = mem::transmute::<u64, [u8; 8]>(word);
            //     for ii in 0..8 {
            //         let b = word_bytes[ii];
            //         if b != 0 {
            //             let ix = b - 1;
            //             return Some((
            //                 i * KEYS_PER_WORD + ii,
            //                 &self.node.ptrs[ix as usize] as *const _ as *mut _,
            //             ));
            //         }
            //     }
            // }
            // unreachable!()
        }

        #[inline(always)]
        fn state_valid(&self) {
            #[cfg(debug_assertions)]
            {
                let mut present = 0;
                for ix in &self.node.keys[..] {
                    present += if *ix > 0 { 1 } else { 0 }
                }
                assert_eq!(
                    present, self.children,
                    "Only see {:?} present but have {:?} children",
                    present, self.children
                )
            }
        }
    }

    impl<T: Element> Node<T> for RawNode<Node48<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            self.state_valid();
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
            self.state_valid();
            unsafe {
                self.get_min_inner()
                    .and_then(|(_, t)| match (*t).get().unwrap() {
                        Ok(t) => Some(t),
                        Err(inner_node) => with_node!(inner_node, node, { node.get_min() }),
                    })
            }
        }
        fn is_full(&self) -> bool {
            self.children == 48
        }
        unsafe fn delete(&mut self, d: u8) -> DeleteResult<T> {
            self.state_valid();
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
                        let (ix, or_ptr) = self.get_min_inner().expect("Should be one more child");
                        self.node.keys[ix] = 0; // not really necessary
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

        unsafe fn insert(
            &mut self,
            d: u8,
            ptr: ChildPtr<T>,
            pptr: Option<*mut ChildPtr<T>>,
        ) -> Result<(), ChildPtr<T>> {
            debug_assert!(self.find_raw(d).is_none());
            self.state_valid();
            debug_assert!(self.children <= 48);
            if self.children == 48 {
                if let Some(pp) = pptr {
                    let new_node = &mut *Box::into_raw(Box::new(RawNode {
                        typ: NODE_256,
                        children: 48,
                        count: self.count,
                        consumed: self.consumed,
                        prefix: self.prefix,
                        node: Node256 {
                            ptrs: mem::transmute::<[usize; 256], [ChildPtr<T>; 256]>(
                                [0 as usize; 256],
                            ),
                        },
                    }));
                    for i in 0..256 {
                        if let Some(node_ptr) = self.find_raw(i as u8) {
                            debug_assert!(i != d as usize, "{:?} == {:?}", i, d);
                            mem::swap(&mut *node_ptr, new_node.node.ptrs.get_unchecked_mut(i))
                        }
                    }
                    let new_cptr = ChildPtr::from_node(new_node);
                    *pp = new_cptr;
                    let res = new_node.insert(d, ptr, None);
                    debug_assert!(res.is_ok());
                    return Ok(());
                } else {
                    return Err(ptr);
                }
            }
            for i in 0..48 {
                let slot = self.node.ptrs.get_unchecked_mut(i);
                if slot.is_null() {
                    ptr::write(slot, ptr);
                    self.node.keys[d as usize] = i as u8 + 1;
                    self.children += 1;
                    return Ok(());
                }
            }
            unreachable!()
        }

        fn for_each<F: FnMut(&T)>(
            &self,
            f: &mut F,
            mut lower: Option<&[u8]>,
            mut upper: Option<&[u8]>,
            lval: Option<&T::Key>,
            rval: Option<&T::Key>,
        ) {
            let low = advance_or(&mut lower, 0);
            let high = advance_or(&mut upper, 255);

            for i in low..(high + 1) {
                let ix = self.node.keys[i];
                if ix == 0 {
                    continue;
                }
                visit_leaf(
                    &self.node.ptrs[ix as usize - 1],
                    f,
                    if i == low { lower } else { None },
                    if i == high { upper } else { None },
                    lval,
                    rval,
                );
            }
        }
    }

    pub struct Node256<T> {
        ptrs: [ChildPtr<T>; 256],
    }

    impl<T> ::std::fmt::Debug for Node256<T> {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
            let mut ix = 0;
            let v: Vec<_> = self.ptrs
                .iter()
                .map(|cp| {
                    let res = (ix, cp);
                    ix += 1;
                    res
                })
                .collect();
            write!(f, "Node256({:?})", v)
        }
    }

    impl<T: Element> Node<T> for RawNode<Node256<T>> {
        fn find_raw(&self, d: u8) -> Option<*mut ChildPtr<T>> {
            unsafe {
                let p = self.node.ptrs.get_unchecked(d as usize) as *const ChildPtr<T>
                    as *mut ChildPtr<T>;
                if (*p).is_null() {
                    None
                } else {
                    Some(p)
                }
            }
        }

        fn is_full(&self) -> bool {
            self.children == 256
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
                    return DeleteResult::Singleton {
                        deleted: deleted,
                        orphan: node.swap_null(),
                    };
                }
                panic!("Should have found a node!")
            }
            DeleteResult::Success(deleted)
        }

        unsafe fn insert(
            &mut self,
            d: u8,
            ptr: ChildPtr<T>,
            _p: Option<*mut ChildPtr<T>>,
        ) -> Result<(), ChildPtr<T>> {
            debug_assert!(self.find_raw(d).is_none(), "d={:?} IN {:?}", d, self);
            debug_assert!(self.children <= 256);
            debug_assert!(self.node.ptrs[d as usize].is_null());
            self.children += 1;
            ptr::write(self.node.ptrs.get_unchecked_mut(d as usize), ptr);
            Ok(())
        }

        fn for_each<F: FnMut(&T)>(
            &self,
            f: &mut F,
            mut lower: Option<&[u8]>,
            mut upper: Option<&[u8]>,
            lval: Option<&T::Key>,
            rval: Option<&T::Key>,
        ) {
            let low = advance_or(&mut lower, 0);
            let high = advance_or(&mut upper, 255);
            for i in low..(high + 1) {
                visit_leaf(
                    &self.node.ptrs[i],
                    f,
                    if i == low { lower } else { None },
                    if i == high { upper } else { None },
                    lval,
                    rval,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rand;
    use super::super::rand::Rng;
    use std::fmt::{Debug, Error, Formatter};

    fn random_vec(max_val: u64, len: usize) -> Vec<u64> {
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen_range::<u64>(0, max_val)).collect()
    }

    fn random_string_vec(max_len: usize, len: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();
        (0..len)
            .map(|_| {
                let s_len = rng.gen_range::<usize>(0, max_len);
                String::from_utf8((0..s_len).map(|_| rng.gen_range::<u8>(0, 128)).collect())
                    .unwrap()
            })
            .collect()
    }

    struct DebugVal<V: Debug + for<'a> Digital<'a>>(V);
    impl<V: Debug + for<'a> Digital<'a>> Debug for DebugVal<V> {
        fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
            write!(
                f,
                "[{:?} : {:?}]",
                self.0,
                self.0.digits().collect::<Vec<_>>()
            )
        }
    }

    #[test]
    fn basic_set_behavior() {
        let mut s = ARTSet::<u64>::new();
        let mut v1 = random_vec(!0, 1 << 18);
        for item in v1.iter() {
            s.add(*item);
            assert!(s.contains(item));
        }
        let mut missing = Vec::new();
        for item in v1.iter() {
            if !s.contains(item) {
                missing.push(*item)
            }
        }
        let v: Vec<_> = missing
            .iter()
            .map(|x| {
                let v: Vec<_> = x.digits().collect();
                (x, v)
            })
            .collect();
        assert_eq!(missing.len(), 0, "missing={:?}", v);
        v1.sort();
        v1.dedup_by_key(|x| *x);
        let mut v2 = Vec::new();
        for _ in 0..(1 << 17) {
            if let Some(x) = v1.pop() {
                v2.push(x)
            } else {
                break;
            }
        }
        for i in v2.iter() {
            s.remove(i);
            assert!(
                !s.contains(i),
                "Deletion failed immediately for {:?}",
                DebugVal(*i)
            );
        }
        let mut failed = false;
        for i in v2.iter() {
            if s.contains(i) {
                eprintln!("Deleted {:?}, but it's still there!", DebugVal(*i));
                failed = true;
            };
        }
        assert!(!failed);
        for i in v1.iter() {
            assert!(s.contains(i), "Didn't delete {:?}, but it is gone!", *i);
        }
    }

    #[test]
    fn string_set_behavior() {
        let mut s = ARTSet::<String>::new();
        let mut v1 = random_string_vec(64, 1 << 18);
        for item in v1.iter() {
            s.add(item.clone());
            assert!(s.contains(item));
        }
        let mut missing = Vec::new();
        for item in v1.iter() {
            if !s.contains(item) {
                missing.push(item.clone())
            }
        }
        let v: Vec<_> = missing
            .iter()
            .map(|x| {
                let v: Vec<_> = x.digits().collect();
                (x, v)
            })
            .collect();
        assert_eq!(missing.len(), 0, "missing={:?}", v);
        v1.sort();
        v1.dedup_by_key(|x| x.clone());
        let mut v2 = Vec::new();
        for _ in 0..(1 << 17) {
            if let Some(x) = v1.pop() {
                v2.push(x)
            } else {
                break;
            }
        }
        for i in v2.iter() {
            s.remove(i);
            assert!(
                !s.contains(i),
                "Deletion failed immediately for {:?}",
                DebugVal(i.clone())
            );
        }
        let mut failed = false;
        for i in v2.iter() {
            if s.contains(i) {
                eprintln!("Deleted {:?}, but it's still there!", DebugVal(i.clone()));
                failed = true;
            };
        }
        assert!(!failed);
        for i in v1.iter() {
            assert!(
                s.contains(i),
                "Didn't delete {:?}, but it is gone!",
                i.clone()
            );
        }
    }

    fn assert_lists_equal<T: Debug + Eq + for<'a> Digital<'a> + Clone>(v1: &[T], v2: &[T]) {
        if v1 == v2 {
            return;
        }
        eprintln!("v1.len()={:?} v2.len()={:?}", v1.len(), v2.len());
        let mut ix = 0;
        for (i, j) in v1.iter().zip(v2.iter()) {
            if *i != *j {
                eprintln!(
                    "[{:4?}] {:20?} != {:20?}",
                    ix,
                    DebugVal(i.clone()),
                    DebugVal(j.clone())
                );
            }
            ix += 1;
        }
        assert!(false, "See error logs");
    }

    #[test]
    fn iterator_behavior() {
        let mut s = ARTSet::<u64>::new();
        let mut v1 = random_vec(!0, 1 << 10);
        for item in v1.iter() {
            s.add(*item);
            assert!(s.contains(item));
        }

        v1.sort();
        v1.dedup_by_key(|x| *x);
        // Iterating over the entire set should give us back the elements in sorted order.
        let mut elts = Vec::new();
        s.for_each_range(|x| elts.push(*x), None, None);
        assert_lists_equal(&v1[..], &elts[..]);
        if v1.len() < 4 {
            // extremely unlikely but retry in this case!
            iterator_behavior();
            return;
        }

        let q1 = v1.len() / 4;
        let q3 = 3 * (v1.len() / 4);
        elts.clear();
        eprintln!("q1={:?} q3={:?}", DebugVal(v1[q1]), DebugVal(v1[q3]));
        s.for_each_range(|x| elts.push(*x), Some(&v1[q1]), Some(&v1[q3]));
        assert_lists_equal(&v1[q1..q3], &elts[..]);
        elts.clear();
        s.for_each_range(|x| elts.push(*x), Some(&v1[q3]), Some(&v1[q1]));
        assert_eq!(elts.len(), 0);
        elts.clear();
        s.for_each_range(|x| elts.push(*x), Some(&v1[q1]), None);
        assert_lists_equal(&v1[q1..], &elts[..]);
        elts.clear();
        s.for_each_range(|x| elts.push(*x), None, Some(&v1[q3]));
        assert_lists_equal(&v1[..q3], &elts[..]);
    }
}
