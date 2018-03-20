#[macro_use]
extern crate criterion;
extern crate radix_tree;
extern crate rand;

use criterion::{Bencher, Criterion, Fun};
use rand::Rng;
use std::collections::btree_set::BTreeSet;
use std::collections::HashSet;
use std::hash::Hash;

use radix_tree::{ARTSet, Digital};

/// Barebones set trait to abstract over various collections.
trait Set<T> {
    fn new() -> Self;
    fn contains(&self, t: &T) -> bool;
    fn insert(&mut self, t: T);
    fn delete(&mut self, t: &T) -> bool;
}

impl<T: for<'a> Digital<'a> + Ord> Set<T> for ARTSet<T> {
    fn new() -> Self {
        ARTSet::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.contains(t)
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

impl<T: Hash + Eq> Set<T> for HashSet<T> {
    fn new() -> Self {
        HashSet::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.get(t).is_some()
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

impl<T: Ord> Set<T> for BTreeSet<T> {
    fn new() -> Self {
        BTreeSet::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.get(t).is_some()
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

fn bench_set_rand_int_lookup_in_set<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.contains(&elts[ix]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn bench_set_rand_int_lookup_not_in_set<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    let elts2: Vec<u64> = (0..set_size).map(|_| rng.gen_range::<u64>(0, !0)).collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.contains(&elts2[ix]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn bench_set_rand_int_insert_remove<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.insert(elts[ix]);
        s.delete(&elts[(ix + 2) & (set_size - 1)]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    fn make_fns<S: Set<u64>>() -> Vec<Fun<()>> {
        vec![
            Fun::new("16k_lookup_hit", |b, _| {
                bench_set_rand_int_lookup_in_set::<S>(b, 16 << 10, !0)
            }),
            Fun::new("1M_lookup_hit", |b, _| {
                bench_set_rand_int_lookup_in_set::<S>(b, 1 << 20, !0)
            }),
            Fun::new("16k_lookup_miss", |b, _| {
                bench_set_rand_int_lookup_not_in_set::<S>(b, 16 << 10, !0)
            }),
            Fun::new("1M_lookup_miss", |b, _| {
                bench_set_rand_int_lookup_not_in_set::<S>(b, 1 << 20, !0)
            }),
            Fun::new("16k_insert_remove", |b, _| {
                bench_set_rand_int_insert_remove::<S>(b, 16 << 10, !0)
            }),
            Fun::new("1M_insert_remove", |b, _| {
                bench_set_rand_int_insert_remove::<S>(b, 1 << 20, !0)
            }),
        ]
    }
    c.bench_functions("Hashtable", make_fns::<HashSet<u64>>(), ());
    c.bench_functions("BTree", make_fns::<BTreeSet<u64>>(), ());
    c.bench_functions("ART", make_fns::<ARTSet<u64>>(), ());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
