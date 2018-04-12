use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use super::common::Digital;

extern crate simd;

#[cfg(target_arch = "x86")]
use std::arch::x86::_mm_movemask_epi8;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_movemask_epi8;
use super::smallvec::{Array, SmallVec};

pub const PREFIX_LEN: usize = 8;
pub type RawMutRef<'a, T> = &'a mut RawNode<T>;
pub type RawRef<'a, T> = &'a RawNode<T>;
pub struct MarkedPtr<T>(usize, PhantomData<T>);
pub use self::node_variants::{NODE_16, NODE_256, NODE_4, NODE_48, Node16, Node256, Node48,
                              NodeType};

pub trait Element {
    type Key: for<'a> Digital<'a> + PartialOrd;
    fn key(&self) -> &Self::Key;
    fn matches(&self, k: &Self::Key) -> bool;
    fn replace_matching(&mut self, other: &mut Self);
}

impl<T> Clone for MarkedPtr<T> {
    fn clone(&self) -> Self {
        MarkedPtr(self.0, PhantomData)
    }
}
pub struct ChildPtr<T>(MarkedPtr<T>);

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
    pub fn null() -> Self {
        ChildPtr(MarkedPtr::null())
    }

    pub fn from_node<R>(p: *mut RawNode<R>) -> Self {
        ChildPtr(MarkedPtr::from_node(p))
    }

    pub fn from_leaf(p: *mut T) -> Self {
        ChildPtr(MarkedPtr::from_leaf(p))
    }

    pub fn swap_null(&mut self) -> Self {
        let mut self_ptr = ChildPtr::null();
        mem::swap(self, &mut self_ptr);
        self_ptr
    }

    pub unsafe fn to_marked(&self) -> MarkedPtr<T> {
        ptr::read(&self.0)
    }
}

impl<T> MarkedPtr<T> {
    pub fn null() -> Self {
        MarkedPtr(0, PhantomData)
    }

    fn from_node<R>(p: *mut RawNode<R>) -> Self {
        debug_assert!(!p.is_null());
        MarkedPtr(p as usize, PhantomData)
    }

    pub fn from_leaf(p: *mut T) -> Self {
        debug_assert!(!p.is_null());
        MarkedPtr((p as usize) | 1, PhantomData)
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    pub fn raw_eq(&self, other: usize) -> bool {
        self.0 == other
    }

    pub unsafe fn get(&self) -> Option<Result<&T, &RawNode<()>>> {
        if self.0 == 0 {
            None
        } else if self.0 & 1 == 1 {
            Some(Ok(&*((self.0 & !1) as *const T)))
        } else {
            Some(Err(&*(self.0 as *const RawNode<()>)))
        }
    }

    pub unsafe fn get_raw(&self) -> Option<Result<*mut T, *mut RawNode<()>>> {
        if self.0 == 0 {
            None
        } else if self.0 & 1 == 1 {
            Some(Ok((self.0 & !1) as *mut T))
        } else {
            Some(Err(self.0 as *mut RawNode<()>))
        }
    }

    pub unsafe fn get_mut(&mut self) -> Option<Result<&mut T, &mut RawNode<()>>> {
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

#[repr(C)]
#[derive(Debug)]
pub struct RawNode<Footer> {
    pub typ: NodeType,
    children: u16,
    pub count: u32,
    pub consumed: u32,
    pub prefix: [u8; PREFIX_LEN],
    node: Footer,
}

impl<T> RawNode<T> {
    pub fn append_prefix(&mut self, d: &[u8], total_count: u32) {
        debug_assert!(d.len() <= PREFIX_LEN);
        unsafe {
            ptr::copy(
                &self.prefix[0],
                &mut self.prefix[d.len()],
                PREFIX_LEN - d.len(),
            );
            ptr::copy(&d[0], &mut self.prefix[0], d.len());
        }
        self.count += total_count;
        self.consumed -= total_count;
    }
}

impl RawNode<()> {
    pub fn get_matching_prefix<T: Element>(
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
    pub fn prefix_matches_optimistic(&self, digits: &[u8]) -> Option<(bool, usize)> {
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

pub enum DeleteResult<T> {
    Failure,
    Success(ChildPtr<T>),
    Singleton {
        deleted: ChildPtr<T>,
        last: ChildPtr<T>,
        last_d: u8,
    },
}

pub trait Node<T: Element>: Sized {
    // insert assumes that 'd' is not present in the node. This is enforced in debug buids
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

    // iterate over all non-null direct children of the node.
    fn local_foreach<F: FnMut(u8, MarkedPtr<T>)>(&self, f: F);

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

pub fn make_node_with_prefix<T>(prefix: &[u8], consumed: u32) -> Box<RawNode<Node4<T>>> {
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

pub fn make_node_from_common_prefix<T>(
    d1: &[u8],
    d2: &[u8],
    consumed: u32,
) -> Box<RawNode<Node4<T>>> {
    let mut common_prefix_digits = SmallVec::<[u8; 32]>::new();
    get_matching_prefix_slice(d1.iter(), d2.iter(), &mut common_prefix_digits);
    make_node_with_prefix(&common_prefix_digits[..], consumed)
}

pub struct Node4<T> {
    keys: [u8; 4],
    ptrs: [ChildPtr<T>; 4],
}

impl<T> ::std::fmt::Debug for Node4<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "Node4({:?}, {:?})", self.keys, &self.ptrs[..])
    }
}

pub fn visit_leaf<T, F>(
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
                        debug_assert!(!$slf.node.ptrs[0].is_null(), "{:?} Uh oh! {:?}", $slf as *const _, $slf);
                        mem::swap(&mut $slf.node.ptrs[0], &mut c_ptr);
                        DeleteResult::Singleton {
                            deleted: deleted,
                            last: c_ptr,
                            last_d: $slf.node.keys[0],
                        }
                    } else {
                        DeleteResult::Success(deleted)
                    }
                }
            }
        };
    }
    impl<T> ::std::fmt::Debug for Node16<T> {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
            write!(f, "Node4({:?}, {:?})", self.keys, &self.ptrs[..])
        }
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

    macro_rules! n416_local_foreach {
        ($slf: expr, $f: expr) => {{
            debug_assert!(is_sorted(&$slf.node.keys[..$slf.children as usize]));
            let children = $slf.children as usize;
            for i in 0..children {
                let k = $slf.node.keys[i];
                let ptr = &$slf.node.ptrs[i];
                debug_assert!(!ptr.is_null());
                $f(k, unsafe { ptr.to_marked() });
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

        fn local_foreach<F: FnMut(u8, MarkedPtr<T>)>(&self, mut f: F) {
            n416_local_foreach!(self, f)
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

        fn local_foreach<F: FnMut(u8, MarkedPtr<T>)>(&self, mut f: F) {
            n416_local_foreach!(self, f)
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

        fn local_foreach<F: FnMut(u8, MarkedPtr<T>)>(&self, mut f: F) {
            for d in 0..256 {
                let i = self.node.keys[d];
                if i == 0 { continue; }
                let ix = i as usize - 1;
                let ptr = &self.node.ptrs[ix];
                unsafe {
                    debug_assert!(!ptr.is_null());
                    debug_assert!(d < 256);
                    f(d as u8, ptr.to_marked())
                };
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
                            last: (*or_ptr).swap_null(),
                            last_d: ix as u8,
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

        fn local_foreach<F: FnMut(u8, MarkedPtr<T>)>(&self, mut f: F) {
            for d in 0..256 {
                unsafe {
                    let ptr =  self.node.ptrs.get_unchecked(d);
                    if ptr.is_null() {
                        continue;
                    }
                    f(d as u8, ptr.to_marked());
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
                    return DeleteResult::Singleton {
                        deleted: deleted,
                        last: node.swap_null(),
                        last_d: i as u8,
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
