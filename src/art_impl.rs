//! Single-threaded radix tree implementation based on HyPer's ART
use std::borrow::Borrow;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use super::Digital;
use super::art_internal::*;
use super::prefix_cache::{HashBuckets, HashSetPrefixCache, NullBuckets};
use super::smallvec::SmallVec;
pub use super::prefix_cache::PrefixCache;

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

pub type ARTSet<T> = RawART<ArtElement<T>, NullBuckets<ArtElement<T>>>;
pub type MidARTSet<T> = RawART<ArtElement<T>, HashSetPrefixCache<ArtElement<T>>>;
pub type LargeARTSet<T> = RawART<ArtElement<T>, HashBuckets<ArtElement<T>>>;

impl<T: for<'a> Digital<'a> + PartialOrd, C: PrefixCache<ArtElement<T>>> RawART<ArtElement<T>, C> {
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

enum PartialResult<T> {
    Failure(T),
    Replaced(T),
    Success,
}

enum PartialDeleteResult<T> {
    Partial,
    Failure,
    Success(T),
}

// TODO:
// 1 Use full predicate to avoid unnecessary heap allocation [check]
// 2 For insertions that create an interior node (case 3 and case 4),
//   need to insert node into hash table. This can be done by passing a
//   reference to the function. [check]
// 3 For removals, any place that removes an interior node (basically
//   the case when a last is returned) check the old nodes `consumed`
//   value and remove it from the table if necessary. [half-done]
// 4 Clean this shit up!

pub struct RawART<T: Element, C: PrefixCache<T>> {
    len: usize,
    root: ChildPtr<T>,
    prefix_target: usize,
    buckets: C,
}

impl<T: Element, C: PrefixCache<T>> RawART<T, C> {
    pub fn new() -> Self {
        RawART::with_prefix_buckets(3, 256 * 256 * 256 * 2)
    }

    pub fn with_prefix_buckets(prefix_len: usize, buckets: usize) -> Self {
        assert!(prefix_len <= 8);
        RawART {
            len: 0,
            root: ChildPtr::null(),
            buckets: C::new(buckets),
            prefix_target: prefix_len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn hash_lookup(&self, digits: &[u8]) -> (bool, Option<MarkedPtr<T>>) {
        // TODO this has to be modified to signal whether digits has a prefix that is too short, or
        // if the lookup failed.
        //
        // One alternative here is to see if digits can be padded out, or disallowed
        // - padded out with stop character for things like string
        // - disallowed for u64, because all "digits" slices are of the same length
        if digits.len() <= self.prefix_target {
            (false, None)
        } else {
            (true, self.buckets.lookup(&digits[0..self.prefix_target]))
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
            mut consumed: usize,
            dont_check: bool,
        ) -> Option<*mut T> {
            match curr.get_raw() {
                None => None,
                Some(Ok(leaf_node)) => {
                    if (dont_check && digits.len() == consumed) || (*leaf_node).matches(k) {
                        Some(leaf_node)
                    } else {
                        None
                    }
                }
                Some(Err(inner_node)) => {
                    consumed = (*inner_node).consumed as usize;
                    if consumed >= digits.len() {
                        return None;
                    }
                    // handle prefixes now
                    (*inner_node)
                        .prefix_matches_optimistic(&digits[consumed..])
                        .and_then(|(dont_check_new, con)| {
                            consumed += con;
                            // let new_digits = &digits[consumed..];
                            if digits.len() == consumed {
                                // Our digits were entirely consumed, but this is a non-leaf node.
                                // That means our node is not present.
                                return None;
                            }
                            with_node!(&*inner_node, nod, {
                                nod.find_raw(digits[consumed]).and_then(|next_node| {
                                    lookup_raw_recursive(
                                        (&*next_node).to_marked(),
                                        k,
                                        digits,
                                        consumed + 1,
                                        dont_check && dont_check_new,
                                    )
                                })
                            })
                        })
                }
            }
        }
        if C::ENABLED {
            let (elligible, opt) = self.hash_lookup(digits.as_slice());
            let node_ref = if let Some(ptr) = opt {
                ptr
            } else if C::COMPLETE && elligible {
                return None;
            } else {
                self.root.to_marked()
            };
            lookup_raw_recursive(node_ref, k, digits.as_slice(), 0, true)
        } else {
            lookup_raw_recursive(self.root.to_marked(), k, digits.as_slice(), 0, true)
        }
    }

    pub unsafe fn delete_raw(&mut self, k: &T::Key) -> Option<T> {
        // Also, consider hypothesis that promoting last doesn't work, and is leading to failed
        // lookups
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        use self::PartialDeleteResult::*;
        unsafe fn delete_raw_recursive<T: Element, C: PrefixCache<T>>(
            k: &T::Key,
            mut curr: MarkedPtr<T>,
            curr_ptr: Option<&mut ChildPtr<T>>,
            parent: Option<(u8, &mut ChildPtr<T>)>,
            digits: &[u8],
            mut consumed: usize,
            target: usize,
            buckets: &mut C,
            is_root: bool,
            // return the deleted node
        ) -> PartialDeleteResult<T> {
            use self::PartialDeleteResult::*;
            if curr.is_null() {
                return Failure;
            }
            unsafe fn move_val_out<T>(mut cptr: ChildPtr<T>) -> T {
                let res = {
                    // first we read the memory out
                    let r = cptr.get_mut().unwrap().unwrap();
                    ptr::read(r)
                };
                // Now we want to deallocate the memory that once held the element, but we don't
                // want to run its destructor if it has one.
                //
                // XXX There must be a better way to do this. Call deallocate directly?
                use std::mem::ManuallyDrop;
                let cptr2 = mem::transmute::<ChildPtr<T>, ChildPtr<ManuallyDrop<T>>>(cptr);
                mem::drop(cptr2);
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
                                            (Success(move_val_out(deleted)), None)
                                        }
                                        DeleteResult::Singleton {
                                            deleted,
                                            last,
                                            last_d,
                                        } => {
                                            debug_assert!(deleted.get().unwrap().is_ok());
                                            (Success(move_val_out(deleted)), Some((last, last_d)))
                                        }
                                        DeleteResult::Failure => unreachable!(),
                                    }
                                }
                            );
                            if let Some((mut c_ptr, last_d)) = asgn {
                                // we are promoting a "last" so we must increase its prefix
                                // length
                                let mut switch = false;
                                let mut replace = false;
                                let mut ds = SmallVec::<[u8; 8]>::new();
                                {
                                    let pp = parent_ref.get_mut().unwrap().err().unwrap();
                                    if C::ENABLED && pp.consumed as usize <= target
                                        && target <= pp.consumed as usize + pp.count as usize
                                    {
                                        // We want to construct enough context to clear out the
                                        // cache below. Because digits[..] may be too short to fill
                                        // the hash prefix cache, we need to fill in additional
                                        // context from the interior nodes.
                                        //
                                        // In this case, we start the work by filling in the prefix
                                        // not present in 'pp'. Below we do the same for `inner` in
                                        // case it replaces 'pp'.
                                        replace = true;
                                        if digits.len() < target {
                                            for dd in &digits[..pp.consumed as usize] {
                                                ds.push(*dd)
                                            }
                                        }
                                    }
                                    if let Err(inner) = c_ptr.get_mut().unwrap() {
                                        // The "last" node that we are promoting is an interior
                                        // node. As a result, we have to modify its prefix and
                                        // potentially insert it into the prefix cache.
                                        let parent_count = pp.count;
                                        let mut prefix_digits =
                                            SmallVec::<[u8; PREFIX_LEN + 1]>::new();
                                        for dd in &pp.prefix
                                            [..cmp::min(parent_count as usize, PREFIX_LEN)]
                                        {
                                            prefix_digits.push(*dd);
                                        }
                                        prefix_digits.push(last_d);
                                        inner.append_prefix(
                                            prefix_digits.as_slice(),
                                            prefix_digits.len() as u32,
                                        );
                                        debug_assert_eq!(inner.consumed, pp.consumed);
                                        if C::ENABLED && inner.consumed as usize <= target
                                            && target
                                                <= inner.consumed as usize + inner.count as usize
                                        {
                                            switch = true;
                                            if digits.len() < target {
                                                if !replace {
                                                    for dd in &digits[..pp.consumed as usize] {
                                                        ds.push(*dd);
                                                    }
                                                }
                                                for dd in &inner.prefix
                                                    [..cmp::min(inner.count as usize, PREFIX_LEN)]
                                                {
                                                    ds.push(*dd);
                                                }
                                            }
                                        }
                                    }
                                    if C::ENABLED && replace && !switch && digits.len() < target {
                                        for dd in
                                            &pp.prefix[..cmp::min(pp.count as usize, PREFIX_LEN)]
                                        {
                                            ds.push(*dd);
                                        }
                                    }
                                }
                                mem::swap(parent_ref, &mut c_ptr);
                                if C::ENABLED {
                                    let mut d_slice = &digits[..];
                                    if digits.len() < target && (switch || replace) {
                                        debug_assert!(target <= 8);
                                        // need to construct new digits
                                        d_slice = ds.as_slice();
                                    }
                                    if switch {
                                        buckets
                                            .insert(&d_slice[0..target], (*parent_ref).to_marked());
                                        debug_assert!(parent_ref.get().unwrap().is_err());
                                    } else if replace {
                                        buckets.insert(&d_slice[0..target], MarkedPtr::null());
                                    }
                                }
                            }
                            return res;
                        } else {
                            None
                        }
                    } else {
                        return Failure;
                    }
                }
                Err(inner_node) => {
                    consumed = inner_node.consumed as usize;
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
                        return Failure;
                    }
                }
            };
            if let Some((inner_node, matched)) = rest_opts {
                let next_digit = digits[consumed + matched];
                with_node_mut!(&mut *inner_node, node, {
                    if let Some(c_ptr) = node.find_mut(next_digit) {
                        consumed += matched + 1;
                        let marked = c_ptr.to_marked();
                        delete_raw_recursive(
                            k,
                            marked,
                            Some(c_ptr),
                            curr_ptr.map(|x| (next_digit, x)),
                            digits,
                            consumed,
                            target,
                            buckets,
                            false,
                        )
                    } else {
                        Failure
                    }
                })
            } else if let Some(cp) = curr_ptr {
                // we are in the root, set curr to null.
                if !is_root {
                    return Partial;
                }
                let c_ptr = cp.swap_null();
                if C::ENABLED {
                    buckets.insert(&digits[0..target], MarkedPtr::null());
                }
                Success(move_val_out(c_ptr))
            } else {
                Partial
            }
        }
        let mut res = Partial;
        if C::ENABLED {
            let (elligible, opt) = self.hash_lookup(digits.as_slice());
            res = if let Some(ptr) = opt {
                delete_raw_recursive(
                    k,
                    ptr,
                    None,
                    None,
                    &digits[..],
                    0,
                    self.prefix_target,
                    &mut self.buckets,
                    false,
                )
            } else if C::COMPLETE && elligible {
                return None;
            } else {
                Partial
            };
        }
        if let Partial = res {
            let marked_root = self.root.to_marked();
            res = delete_raw_recursive(
                k,
                marked_root,
                Some(&mut self.root),
                None,
                &digits[..],
                0,
                self.prefix_target,
                &mut self.buckets,
                true,
            );
        }
        match res {
            Success(x) => {
                debug_assert!(self.len > 0);
                self.len -= 1;
                Some(x)
            }
            Failure => None,
            Partial => panic!("Got a partial!"),
        }
    }

    pub unsafe fn insert_raw(&mut self, elt: T) -> Result<(), T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(elt.key().digits());

        unsafe fn insert_raw_recursive<T: Element, C: PrefixCache<T>>(
            curr: MarkedPtr<T>,
            mut e: T,
            digits: &[u8],
            mut consumed: usize,
            pptr: Option<*mut ChildPtr<T>>,
            buckets: &mut C,
            target: usize,
        ) -> PartialResult<T> {
            use self::PartialResult::*;
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
                    if C::ENABLED && consumed <= target
                        && target <= consumed + (*n4_raw).count as usize
                    {
                        buckets.insert(&digits[0..target], (*pp).to_marked());
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
                    #[cfg(debug_assertions)]
                    {
                        if pptr.is_some() {
                            debug_assert_eq!(consumed, inner_node.consumed as usize);
                        }
                    }
                    consumed = inner_node.consumed as usize;
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
                            let full = nod.is_full();
                            if C::ENABLED && full && pptr.is_none() {
                                return Failure(e);
                            }
                            let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));
                            let _r = nod.insert(d, c_ptr, pptr);
                            debug_assert!(_r.is_ok());
                            if C::ENABLED && full && nod.consumed as usize <= target
                                && target <= nod.consumed as usize + nod.count as usize
                            {
                                buckets.insert(&digits[0..target], (*pptr.unwrap()).to_marked());
                            }
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
                        if C::ENABLED && consumed <= target
                            && target <= consumed + (*n4_raw).count as usize
                        {
                            buckets.insert(&digits[0..target], (*pp).to_marked());
                            debug_assert!((*pp).get().unwrap().is_err());
                        }
                        (*n4_raw).insert(inner_d, n4_cptr, None).unwrap()
                    }
                }
            };
            Success
        }
        if C::ENABLED {
            let e = {
                let (node_ref, consumed, pptr) = {
                    if let Some(ptr) = self.hash_lookup(digits.as_slice()).1 {
                        (ptr, self.prefix_target, None)
                    } else {
                        let root_alias = Some(&mut self.root as *mut _);
                        (self.root.to_marked(), 0, root_alias)
                    }
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
                    PartialResult::Failure(e) => e,
                    PartialResult::Success => {
                        self.len += 1;
                        return Ok(());
                    }
                    PartialResult::Replaced(t) => {
                        return Err(t);
                    }
                }
            };
            // Hash-indexed inserts can fail, retry a default root-based traversal.
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
                PartialResult::Success => {
                    self.len += 1;
                    Ok(())
                }
                PartialResult::Replaced(t) => Err(t),
                PartialResult::Failure(_) => unreachable!(),
            }
        } else {
            let root_alias = Some(&mut self.root as *mut _);
            match insert_raw_recursive(
                self.root.to_marked(),
                elt,
                digits.as_slice(),
                0,
                root_alias,
                &mut self.buckets,
                self.prefix_target,
            ) {
                PartialResult::Success => {
                    self.len += 1;
                    Ok(())
                }
                PartialResult::Replaced(t) => Err(t),
                PartialResult::Failure(_) => unreachable!(),
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

    macro_rules! for_each_set {
        ($s:ident, $body:expr, $( $base:tt - $param:tt),+) => {
            $({
                let mut $s = $base::<$param>::new();
                $body
            };)+
        };
    }

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
        for_each_set!(
            s,
            {
                let mut v1 = random_vec(!0, 1 << 18);
                for item in v1.iter() {
                    s.add(*item);
                    assert!(
                        s.contains(item),
                        "lookup failed immediately for {:?}",
                        DebugVal(*item)
                    );
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
                let mut failures = 0;
                for i in v2.iter() {
                    let mut fail = 0;
                    if !s.contains(i) {
                        eprintln!("{:?} no longer in the set!", DebugVal(*i));
                        fail = 1;
                    }
                    let res = s.remove(i);
                    if !res {
                        // eprintln!("Deletion failed at call-site for {:?}", DebugVal(*i));
                        fail = 1;
                    }
                    if s.contains(i) {
                        // eprintln!("Deletion failed immediately for {:?}", DebugVal(*i));
                        fail = 1;
                    }
                    failures += fail;
                }
                assert_eq!(failures, 0);
                let mut failed = false;
                for i in v2.iter() {
                    if s.contains(i) {
                        eprintln!("Deleted {:?}, but it's still there!", DebugVal(*i));
                        failed = true;
                    };
                }
                assert!(!failed);
                for i in v1.iter() {
                    assert!(
                        s.contains(i),
                        "Didn't delete {:?}, but it is gone!",
                        DebugVal(*i)
                    );
                }
            },
            ARTSet - u64,
            MidARTSet - u64,
            LargeARTSet - u64
        );
    }

    #[test]
    fn string_set_behavior() {
        for_each_set!(
            s,
            {
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
            },
            ARTSet - String,
            MidARTSet - String,
            LargeARTSet - String
        );
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
