#![feature(swap_nonoverlapping)]
#![feature(cfg_target_feature)]
mod common;
mod st;

pub use common::Digital;
pub use st::*;
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
