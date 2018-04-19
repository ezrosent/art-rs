//! Single-threaded radix tree implementation based on HyPer's ART
use std::borrow::Borrow;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use super::Digital;
use super::art_internal::*;
use super::prefix_cache::{HashSetPrefixCache, NullBuckets};
use super::smallvec::SmallVec;
pub use super::prefix_cache::PrefixCache;

const BAD_DIGITS: [u8; 149] = [245, 135, 131, 129, 244, 178, 172, 139, 242, 161, 188, 142, 244, 163, 158, 175, 241, 145, 165, 178, 241, 180, 160, 189, 242, 188, 173, 170, 241, 167, 131, 136, 242, 172, 180, 186, 241, 150, 165, 135, 241, 187, 168, 190, 243, 179, 185, 136, 241, 186, 149, 136, 243, 139, 165, 153, 242, 168, 163, 174, 243, 165, 185, 185, 244, 144, 177, 184, 243, 168, 130, 176, 230, 171, 179, 229, 151, 180, 243, 138, 152, 139, 243, 192, 140, 186, 241, 168, 167, 163, 241, 162, 137, 134, 242, 148, 142, 163, 241, 181, 138, 151, 244, 143, 181, 185, 244, 144, 133, 131, 243, 161, 151, 177, 241, 146, 159, 175, 241, 166, 166, 129, 242, 183, 180, 188, 244, 135, 149, 168, 243, 184, 158, 134, 243, 144, 161, 157, 227, 141, 180, 225, 187, 143, 243, 174, 186, 165, 0];

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
pub type CachingARTSet<T> = RawART<ArtElement<T>, HashSetPrefixCache<ArtElement<T>>>;

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

pub struct RawART<T: Element, C: PrefixCache<T>> {
    len: usize,
    root: ChildPtr<T>,
    prefix_target: usize,
    buckets: C,
}

impl<T: Element, C: PrefixCache<T>> RawART<T, C> {
    pub fn new() -> Self {
        RawART::with_prefix_buckets(6)
    }

    pub fn with_prefix_buckets(prefix_len: usize) -> Self {
        assert!(prefix_len <= 8);
        assert!(prefix_len > 0);
        RawART {
            len: 0,
            root: ChildPtr::null(),
            buckets: C::new(),
            prefix_target: prefix_len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn hash_lookup(&self, digits: &[u8]) -> (bool, Option<Result<*mut T, MarkedPtr<T>>>) {
        if digits.len() <= self.prefix_target {
            (false, None)
        } else {
            let res = self.buckets.lookup(&digits[0..self.prefix_target]);
            (
                true,
                match res {
                    Some(ptr) => Some({
                        match unsafe { ptr.get_raw().unwrap() } {
                            Ok(leaf) => Ok(leaf),
                            Err(_) => Err(ptr),
                        }
                    }),
                    None => None,
                },
            )
        }
    }

    // replace with NonNull
    pub unsafe fn lookup_raw(&self, k: &T::Key) -> Option<*mut T> {
        let mut digits = SmallVec::<[u8; 32]>::new();
        digits.extend(k.digits());
        let _check = &digits[..] == &BAD_DIGITS[..];
        trace!(_check, "lookup_raw");
        unsafe fn lookup_raw_recursive<T: Element>(
            curr: MarkedPtr<T>,
            k: &T::Key,
            digits: &[u8],
            mut consumed: usize,
            dont_check: bool,
        ) -> Option<*mut T> {
            let _check = digits == &BAD_DIGITS[..];
            match curr.get_raw() {
                None => None,
                Some(Ok(leaf_node)) => {
                    if (dont_check && digits.len() == consumed) || (*leaf_node).matches(k) {
                        trace!(
                            _check,
                            "FOUND dont_check={}, consumed_check={}, matches={}",
                            dont_check,
                            digits.len() == consumed,
                            (*leaf_node).matches(k)
                        );
                        Some(leaf_node)
                    } else {
                        trace!(_check);
                        None
                    }
                }
                Some(Err(inner_node)) => {
                    consumed = (*inner_node).consumed as usize;
                    trace!(_check, "[lookup, d={}] found an inner node {:?}@{:?}", digits[consumed], *inner_node, inner_node);
                    if consumed >= digits.len() {
                        trace!(_check, "consumed too big");
                        return None;
                    }
                    // handle prefixes now
                    (*inner_node)
                        .prefix_matches_optimistic(&digits[consumed..])
                        .and_then(|(dont_check_new, con)| {
                            consumed += con;
                            // let new_digits = &digits[consumed..];
                            if digits.len() == consumed {
                                trace!(_check);
                                // Our digits were entirely consumed, but this is a non-leaf node.
                                // That means our node is not present.
                                return None;
                            }
                            with_node!(&*inner_node, nod, {
                                nod.find_raw(digits[consumed]).and_then(|next_node| {
                                    trace!(_check);
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
            trace!(_check);
            let (elligible, opt) = self.hash_lookup(digits.as_slice());
            let node_ref = if let Some(ptr) = opt {
                match ptr {
                    Ok(leaf) => {
                        return if (*leaf).matches(k) {
                            trace!(_check);
                            Some(leaf)
                        } else {
                            trace!(_check);
                            None
                        }
                    }
                    Err(node) => node,
                }
            } else if C::COMPLETE && elligible && self.len > 1 {
                trace!(_check);
                return None;
            } else {
                trace!(_check);
                self.root.to_marked()
            };
            trace!(_check);
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
        let _check = &digits[..] == &BAD_DIGITS[..];
        trace!(_check, "delete_raw {:?}", &digits[..]);
        unsafe fn delete_raw_recursive<T: Element, C: PrefixCache<T>>(
            k: &T::Key,
            mut curr: MarkedPtr<T>,
            curr_ptr: Option<&mut ChildPtr<T>>,
            parent: Option<(u8, Result<MarkedPtr<T>, &mut ChildPtr<T>>)>,
            digits: &[u8],
            mut consumed: usize,
            target: usize,
            buckets: &mut C,
            is_root: bool,
            // return the deleted node
        ) -> PartialDeleteResult<T> {
            let _check = digits == &BAD_DIGITS[..];
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
                let cptr2 = mem::transmute::<ChildPtr<T>, ChildPtr<mem::ManuallyDrop<T>>>(cptr);
                mem::drop(cptr2);
                res
            }

            let rest_opts = match curr.get_mut().unwrap() {
                Ok(leaf_node) => {
                    trace!(_check);
                    if leaf_node.matches(k) {
                        trace!(_check);
                        // we have a match! delete the leaf
                        if let Some((d, mut parent_ref)) = parent {
                            let (res, asgn) = with_node_mut!(
                                match parent_ref {
                                    Ok(ref mut marked_parent) => {
                                        let p_ref = marked_parent.get_mut().unwrap().err().unwrap();
                                        trace!(_check, "{:?}", p_ref);
                                        if p_ref.children == 2 {
                                            trace!(_check, "[delete] returning partial");
                                            return Partial;
                                        }
                                        p_ref
                                    }
                                    Err(ref mut parent_ptr) => {
                                        trace!(_check);
                                        parent_ptr.get_mut().unwrap().err().unwrap()
                                    }
                                },
                                node,
                                {
                                    trace!(_check, "digits={:?}, d={}", digits, d);
                                    match node.delete(d) {
                                        DeleteResult::Success(deleted) => {
                                            // we are deleteing an individual node. Time to check
                                            // if it is in buckets: if it is we should remove it.
                                            if C::ENABLED && digits.len() >= target
                                                && consumed <= target
                                            {
                                                trace!(_check);
                                                if C::COMPLETE {
                                                    debug_assert!(
                                                        buckets
                                                            .lookup(&digits[0..target])
                                                            .is_some()
                                                    );
                                                }
                                                buckets
                                                    .insert(&digits[0..target], MarkedPtr::null());
                                            }
                                            trace!(_check);
                                            (Success(move_val_out(deleted)), None)
                                        }
                                        DeleteResult::Singleton {
                                            deleted,
                                            last,
                                            last_d,
                                        } => {
                                            trace!(_check);
                                            if C::ENABLED && digits.len() >= target
                                                && consumed <= target
                                            {
                                                trace!(_check);
                                                if C::COMPLETE {
                                                    debug_assert!(
                                                        buckets
                                                            .lookup(&digits[0..target])
                                                            .is_some()
                                                    );
                                                }
                                                buckets
                                                    .insert(&digits[0..target], MarkedPtr::null());
                                            }
                                            if C::ENABLED {
                                                trace!(_check);
                                                if let Ok(_leaf) = last.get().unwrap() {
                                                    let mut leaf_digits =
                                                        SmallVec::<[u8; 8]>::new();
                                                    let leaf: &T = _leaf;
                                                    leaf_digits.extend(leaf.key().digits().take(8));
                                                    if leaf_digits.len() >= target
                                                        && consumed <= target
                                                    {
                                                        trace!(_check);
                                                        buckets.insert(
                                                            &leaf_digits[0..target],
                                                            last.to_marked(),
                                                        );
                                                        debug_assert_eq!(
                                                            buckets.lookup(&leaf_digits[0..target]),
                                                            Some(last.to_marked())
                                                        );
                                                        // N.B. when debugging deletes, consider
                                                        // this extra consistency check. This is
                                                        // off by default because it does an O(n)
                                                        // scan of `buckets` which slows things
                                                        // down considerably on debug builds.
                                                        //
                                                        // // this declaration needs to be moved up
                                                        // // a few blocks
                                                        // let marked_p = parent_ref.to_marked();
                                                        // eprintln!("Remapping digits {:?} while deleting {:?}",
                                                        //           &leaf_digits[..], &digits[..]);
                                                        // buckets.debug_assert_unreachable(marked_p);
                                                    }
                                                }
                                            }
                                            debug_assert!(deleted.get().unwrap().is_ok());
                                            (Success(move_val_out(deleted)), Some((last, last_d)))
                                        }
                                        DeleteResult::Failure => unreachable!(),
                                    }
                                }
                            );
                            if let Some((mut c_ptr, last_d)) = asgn {
                                let _check_2 = &digits[0..3] == &BAD_DIGITS[0..3];
                                trace!(_check);
                                // we are promoting a "last" so we must increase its prefix
                                // length
                                let mut switch = false; // flag for inserting a new interior node

                                // flag for invalidating the cache (as it may contain the node we are deleting)
                                let mut replace = false;
                                let mut ds = SmallVec::<[u8; 8]>::new();
                                {
                                    let _p_marked = match parent_ref {
                                        Ok(ref m) => m.clone(),
                                        Err(ref ptr) => ptr.to_marked(),
                                    };
                                    let pp = match parent_ref {
                                        Ok(ref m) => m.get().unwrap().err().unwrap(),
                                        Err(ref ptr) => ptr.get().unwrap().err().unwrap(),
                                    };
                                    if C::ENABLED && pp.consumed as usize <= target
                                        && target <= pp.consumed as usize + pp.count as usize
                                    {
                                        trace!(_check);
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
                                            for dd in &digits[0..pp.consumed as usize] {
                                                ds.push(*dd)
                                            }
                                        }
                                    } else if C::ENABLED && digits.len() >= target {
                                        trace!(_check);
                                        debug_assert!(
                                            buckets.lookup(&digits[0..target]) != Some(_p_marked)
                                        );
                                    }
                                    if let Err(inner) = c_ptr.get_mut().unwrap() {
                                        let from = format!("{:?}", inner);
                                        // The "last" node that we are promoting is an interior
                                        // node. As a result, we have to modify its prefix and
                                        // potentially insert it into the prefix cache.
                                        let parent_count = pp.count;
                                        let mut prefix_digits =
                                            SmallVec::<[u8; PREFIX_LEN + 1]>::new();
                                        for dd in &pp.prefix
                                            [0..cmp::min(parent_count as usize, PREFIX_LEN)]
                                        {
                                            prefix_digits.push(*dd);
                                        }
                                        prefix_digits.push(last_d);
                                        inner.append_prefix(
                                            prefix_digits.as_slice(),
                                            prefix_digits.len() as u32,
                                        );
                                        trace!(_check_2, "[last_d={}] updating inner node @{:?} {} => {:?} (min={:?})",
                                               last_d, inner as *const _, from, inner,
                                               with_node!(inner, nod, nod.get_min().unwrap().key().digits().collect::<Vec<u8>>(), T));
                                        debug_assert_eq!(inner.consumed, pp.consumed);
                                        if C::ENABLED && inner.consumed as usize <= target
                                            && target
                                                <= inner.consumed as usize + inner.count as usize
                                        {
                                            trace!(_check);
                                            switch = true;
                                            if digits.len() < target {
                                                if !replace {
                                                    trace!(_check);
                                                    for dd in &digits[0..pp.consumed as usize] {
                                                        ds.push(*dd);
                                                    }
                                                }
                                                for dd in &inner.prefix
                                                    [0..cmp::min(inner.count as usize, PREFIX_LEN)]
                                                {
                                                    trace!(_check);
                                                    ds.push(*dd);
                                                }
                                            }
                                        }
                                    }
                                    if C::ENABLED && replace && !switch && digits.len() < target {
                                        trace!(_check);
                                        for dd in
                                            &pp.prefix[0..cmp::min(pp.count as usize, PREFIX_LEN)]
                                        {
                                            ds.push(*dd);
                                        }
                                    }
                                }
                                let c_marked = c_ptr.to_marked();
                                mem::swap(parent_ref.err().unwrap(), &mut c_ptr);
                                if C::ENABLED {
                                    trace!(_check);
                                    if switch || replace {
                                        let mut dsn = SmallVec::<[u8; 8]>::new();
                                        let mut d_slice = &digits[..];
                                        if digits.len() < target {
                                            debug_assert!(target <= 8);
                                            // need to construct new digits
                                            d_slice = ds.as_slice();
                                        }
                                        if consumed <= target {
                                            // there's an edge case here. If consumed == target,
                                            // and digits is of lenght >= target, then the promoted
                                            // node will not have the same target-length prefix as
                                            // digits[..]. It will share all but the last element.
                                            for d in &digits[0..target - 1] {
                                                dsn.push(*d)
                                            }
                                            dsn.push(last_d);
                                            d_slice = &dsn[..]
                                        }
                                        trace!(_check);
                                        buckets.insert(&d_slice[0..target], c_marked);
                                    }
                                }
                            }
                            trace!(_check);
                            return res;
                        } else {
                            trace!(_check);
                            None
                        }
                    } else {
                        trace!(_check);
                        return Failure;
                    }
                }
                Err(inner_node) => {
                    #[cfg(debug_assertions)]
                    {
                        with_node!(inner_node, nod, {
                            let _leaf = nod.get_min().unwrap();
                            let mut _leaf_ds = Vec::with_capacity(digits.len());
                            _leaf_ds.extend(_leaf.key().digits());
                            trace!(_check, "[delete, d={}] found an inner node {:?}@{:?}\n\t(leaf_ds={:?})",
                                   digits[nod.consumed as usize],
                                   nod,
                                   inner_node as *const RawNode<()>,
                                   _leaf_ds);
                        }, T);
                    }
                    debug_assert!(
                        inner_node.consumed as usize <= digits.len(),
                        "inner_node.consumed={} too high, nod={:?}",
                        inner_node.consumed,
                        inner_node
                    );
                    consumed = inner_node.consumed as usize;
                    let (matched, _) = inner_node.get_matching_prefix(
                        digits,
                        consumed,
                        PhantomData as PhantomData<T>,
                    );
                    // if the prefix matches, recur, otherwise just bail out
                    if matched == inner_node.count as usize {
                        trace!(_check);
                        // the prefix matched! we recur below
                        debug_assert!(digits.len() > matched);
                        Some((inner_node as *mut RawNode<()>, matched))
                    } else {
                        trace!(_check, "delete failing consumed={}", consumed);
                        // prefix was not a match, the key is not here
                        return Failure;
                    }
                }
            };
            if let Some((inner_node, matched)) = rest_opts {
                trace!(_check);
                let next_digit = digits[consumed + matched];
                with_node_mut!(&mut *inner_node, node, {
                    if let Some(c_ptr) = node.find_mut(next_digit) {
                        trace!(_check);
                        consumed += matched + 1;
                        let marked = c_ptr.to_marked();
                        delete_raw_recursive(
                            k,
                            marked,
                            Some(c_ptr),
                            Some((
                                next_digit,
                                match curr_ptr {
                                    Some(x) => Err(x),
                                    None => Ok(curr),
                                },
                            )),
                            digits,
                            consumed,
                            target,
                            buckets,
                            false,
                        )
                    } else {
                        trace!(_check);
                        Failure
                    }
                })
            } else if let Some(cp) = curr_ptr {
                if !is_root {
                    trace!(_check);
                    return Partial;
                }
                trace!(_check);
                // we are in the root, set curr to null.
                let c_ptr = cp.swap_null();
                if C::ENABLED && digits.len() >= target {
                    buckets.insert(&digits[0..target], MarkedPtr::null());
                }
                Success(move_val_out(c_ptr))
            } else {
                trace!(_check);
                Partial
            }
        }
        let mut res = Partial;
        if C::ENABLED {
            let (elligible, opt) = self.hash_lookup(digits.as_slice());
            res = if let Some(ptr) = opt {
                trace!(_check, "cache hit");
                match ptr {
                    Ok(_leaf) => Partial,
                    Err(inner) => {
                        #[cfg(debug_assertions)]
                        with_node!(inner.get().unwrap().err().unwrap(), node, {
                            let min = node.get_min().unwrap();
                            let mut min_ds = SmallVec::<[u8; 10]>::new();
                            min_ds.extend(min.key().digits());
                            if _check {
                                assert_eq!(&min_ds[0..self.prefix_target-1], &digits[0..self.prefix_target-1]);
                            }
                        }, T);
                        delete_raw_recursive(
                            k,
                            inner,
                            None,
                            None,
                            &digits[..],
                            0,
                            self.prefix_target,
                            &mut self.buckets,
                            false,
                        )
                    },
                }
            } else if C::COMPLETE && elligible && self.len > 1 {
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
                if C::ENABLED && digits.len() >= target && consumed <= target {
                    debug_assert!(buckets.lookup(&digits[0..target]).is_none());
                    buckets.insert(&digits[0..target], (*pptr.unwrap()).to_marked());
                }

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
                    let mut leaf_digits = SmallVec::<[u8; 8]>::new();
                    leaf_digits.extend(leaf_node.key().digits());
                    let pp = pptr.unwrap();
                    let n4: Box<RawNode<Node4<T>>> = make_node_from_common_prefix(
                        &leaf_digits[consumed..],
                        &digits[consumed..],
                        consumed as u32,
                    );
                    let prefix_len = n4.count as usize;
                    let mut n4_raw = Box::into_raw(n4);
                    let mut leaf_ptr = ChildPtr::from_node(n4_raw);
                    let new_leaf = ChildPtr::from_leaf(Box::into_raw(Box::new(e)));
                    mem::swap(&mut *pp, &mut leaf_ptr);

                    if C::ENABLED && consumed <= target
                        && target <= consumed + (*n4_raw).count as usize
                    {
                        buckets.insert(&digits[0..target], (*pp).to_marked());
                        debug_assert!((*pp).get().unwrap().is_err());
                    } else if C::ENABLED && digits.len() >= target && consumed <= target {
                        debug_assert!(buckets.lookup(&digits[0..target]).is_none());
                        buckets.insert(&digits[0..target], (*pp).to_marked());

                        // buckets.insert(&digits[0..target], new_leaf.to_marked());
                    }

                    if C::ENABLED && leaf_digits.len() >= target && consumed <= target {
                        buckets.insert(&leaf_digits[0..target], (*pp).to_marked());
                    }
                    if C::ENABLED && C::COMPLETE && leaf_digits.len() >= target
                        && consumed <= target
                    {
                        debug_assert!(buckets.lookup(&leaf_digits[0..target]).is_some())
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
                            if C::ENABLED {
                                if nod.consumed as usize <= target
                                    && target <= nod.consumed as usize + nod.count as usize
                                {
                                    if full {
                                        let marked_p = (*pptr.unwrap()).to_marked();
                                        buckets.insert(&digits[0..target], marked_p.clone());
                                    }
                                } else if digits.len() >= target && consumed <= target && !full {
                                    #[cfg(debug_assertions)]
                                    {
                                        if let Some(ptr) = buckets.lookup(&digits[0..target]) {
                                            match ptr.get().unwrap() {
                                                Ok(_leaf) => eprintln!("overwriting leaf node!"),
                                                Err(other_inner) =>
                                                    eprintln!("overwriting inner node: {:?} ptr={:?} pptr={:?} inner={:?}",
                                                              other_inner,
                                                              ptr,
                                                              pptr.map(|x| &*x),
                                                              inn),
                                            }
                                            panic!("Overwriting leaf insertion");
                                        }
                                    }

                                    buckets.insert(&digits[0..target], MarkedPtr::from_node(nod));
                                } else if full && consumed <= target {
                                    let marked_p = (*pptr.unwrap()).to_marked();
                                    // If we were full we have to remap all leaves that are
                                    // children of nod to the new value.
                                    let mut mp = marked_p.clone();
                                    let new_nod = mp.get_mut().unwrap().err().unwrap();
                                    with_node_mut!(
                                        new_nod,
                                        nod,
                                        {
                                            nod.local_foreach(|_, n| {
                                                if let Ok(leaf) = n.get().unwrap() {
                                                    let mut ds = SmallVec::<[u8; 8]>::new();
                                                    ds.extend(leaf.key().digits());
                                                    if ds.len() < target {
                                                        return;
                                                    }
                                                    buckets
                                                        .insert(&ds[0..target], marked_p.clone());
                                                }
                                            });
                                        },
                                        T
                                    );
                                }
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

                        // first make a new node that will be the parent to both `inner` and a leaf
                        // containing `e`.
                        let common_prefix_digits = &digits[consumed..consumed + matched];
                        debug_assert_eq!(common_prefix_digits.len(), matched);
                        let n4: Box<RawNode<Node4<T>>> =
                            make_node_with_prefix(&common_prefix_digits[..], consumed as u32);
                        inner_node.consumed += n4.count + 1;
                        debug_assert_eq!(n4.count as usize, common_prefix_digits.len());
                        let update_cache_inner = C::ENABLED && consumed <= target
                            && target <= consumed + n4.count as usize;
                        consumed += n4.count as usize;
                        let by = matched + 1;
                        adjust_prefix(inner_node, by, min_ref, consumed);

                        // Now allocate a node to contain `e`, insert it into the prefix cache if
                        // necessary, and insert it into n4.
                        let c_ptr = ChildPtr::<T>::from_leaf(Box::into_raw(Box::new(e)));

                        let mut n4_raw = Box::into_raw(n4);
                        let _r = (*n4_raw).insert(digits[consumed], c_ptr, None);
                        debug_assert!(_r.is_ok());
                        let mut n4_cptr = ChildPtr::from_node(n4_raw);
                        // Now swap `inner` with n4 (thereby inserting it into the tree) and insert
                        // `inner` as a child of n4.
                        let pp = pptr.unwrap();
                        mem::swap(&mut *pp, &mut n4_cptr);
                        if update_cache_inner {
                            buckets.insert(&digits[0..target], (*pp).to_marked());
                            debug_assert!((*pp).get().unwrap().is_err());
                        } else if C::ENABLED && digits.len() >= target && consumed <= target {
                            buckets.insert(&digits[0..target], (*pp).to_marked());
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
                    let (_, opt) = self.hash_lookup(digits.as_slice());
                    if let Some(Err(inner)) = opt {
                        (inner, self.prefix_target, None)
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
    // use super::super::rand::Rng;
    // Use StdRng::rom_seed to debug test failures with deterministic inputs
    use super::super::rand::{Rng, SeedableRng, StdRng};
    use std::fmt::{Debug, Error, Formatter};

    macro_rules! for_each_set {
        ($s:ident, $body:expr, $( $base:tt - $param:tt),+) => {
            $({
                // eprintln!("Testing {}", stringify!($base));
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
        const RAND_SEED: [usize; 32] = [1; 32];
        let mut rng = StdRng::from_seed(&RAND_SEED[..]);
        (0..len.next_power_of_two())
            .map(|_| {
                let mlen = max_len as isize;
                let s_len = mlen + rng.gen_range::<isize>(-mlen / 2, mlen / 2);
                rng.gen_iter::<char>()
                    .take(s_len as usize)
                    .collect::<String>()
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
                {
                    let mut i = 0;
                    for item in v1.iter() {
                        s.add(*item);
                        assert!(
                            s.contains(item),
                            "[{:?}] lookup failed immediately for {:?}",
                            i,
                            DebugVal(*item)
                        );
                        i += 1;
                    }
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
                {
                    let mut ix = 0;
                    for i in v2.iter() {
                        let mut fail = 0;
                        if !s.contains(i) {
                            eprintln!("[{}] {:?} no longer in the set!", ix, DebugVal(*i));
                            fail = 1;
                        }
                        let res = s.remove(i);
                        if !res {
                            eprintln!("[{}] Failed to remove {:?}!", ix, DebugVal(*i));
                            fail = 1;
                        }
                        if s.contains(i) {
                            eprintln!(
                                "[{}] {:?} still in the set after removal!",
                                ix,
                                DebugVal(*i)
                            );
                            fail = 1;
                        }
                        failures += fail;
                        ix += 1;
                    }
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
            CachingARTSet - u64,
            ARTSet - u64
        );
    }

    #[test]
    fn string_set_insert_remove() {
        for_each_set!(
            s,
            {
                let v1 = random_string_vec(30, 1 << 18);
                {
                    let mut failed = false;
                    for (i, item) in v1.iter().enumerate() {
                        s.add(item.clone());
                        if !s.contains(item) {
                            failed = true;
                            eprintln!(
                                "[{}] lookup failed immediately for {:?}",
                                i,
                                DebugVal(item.clone())
                            );
                        }
                    }
                    assert!(!failed);
                }
                let mut ix = 0;
                for t in 0..(1 << 18) {
                    s.add(v1[ix].clone());
                    assert!(s.contains(&v1[ix]));
                    ix += 1;
                    ix %= 1 << 18;
                    let in_set = s.contains(&v1[ix]);
                    let deleted = s.remove(&v1[ix]);
                    assert!(!in_set || deleted, "in_set={}, deleted={}", in_set, deleted);
                    assert!(
                        !s.contains(&v1[ix]),
                        "failed assertion (deleted={}) at t={} str={:?}",
                        deleted,
                        t,
                        DebugVal(v1[ix].clone())
                    );
                    ix += 1;
                    ix %= 1 << 18;
                }
            },
            CachingARTSet - String,
            ARTSet - String
        );
    }

    #[test]
    fn string_set_behavior() {
        for_each_set!(
            s,
            {
                let mut v1 = random_string_vec(30, 1 << 18);
                {
                    let mut failed = false;
                    for (i, item) in v1.iter().enumerate() {
                        s.add(item.clone());
                        if !s.contains(item) {
                            failed = true;
                            eprintln!(
                                "[{}] lookup failed immediately for {:?}",
                                i,
                                DebugVal(item.clone())
                            );
                        }
                    }
                    assert!(!failed);
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
                for (t, i) in v2.iter().enumerate() {
                    s.remove(i);
                    assert!(
                        !s.contains(i),
                        "[{}] Deletion failed immediately for {:?}",
                        t, DebugVal(i.clone())
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
            CachingARTSet - String,
            ARTSet - String
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
