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
            _ => panic!("Found unrecognized node type {:?}", _b.typ),
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
