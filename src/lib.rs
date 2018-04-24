#![feature(swap_nonoverlapping)]
#![feature(cfg_target_feature)]
#![feature(stdsimd)]
#[macro_use]
mod macros;
mod common;
mod art_impl;
mod art_internal;
mod prefix_cache;

extern crate byteorder;
extern crate smallvec;

pub use common::Digital;
pub use art_impl::*;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
extern crate rand;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
