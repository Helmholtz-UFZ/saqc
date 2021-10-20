# do not import dios-stuff here
import operator as op


_OP1_MAP = {
    op.inv: "~",
    op.neg: "-",
    op.abs: "abs()",
}

_OP2_COMP_MAP = {
    op.eq: "==",
    op.ne: "!=",
    op.le: "<=",
    op.ge: ">=",
    op.gt: ">",
    op.lt: "<",
}

_OP2_BOOL_MAP = {
    op.and_: "&",
    op.or_: "|",
    op.xor: "^",
}
_OP2_ARITH_MAP = {
    op.add: "+",
    op.sub: "-",
    op.mul: "*",
    op.pow: "**",
}

_OP2_DIV_MAP = {
    op.mod: "%",
    op.truediv: "/",
    op.floordiv: "//",
}

OP_MAP = _OP2_COMP_MAP.copy()
OP_MAP.update(_OP2_BOOL_MAP)
OP_MAP.update(_OP2_ARITH_MAP)
OP_MAP.update(_OP2_DIV_MAP)
OP_MAP.update(_OP1_MAP)
