# Ventris

## Language

**Architecture**:
A Ventris model design, such as Vanilla, RoPE, or MLA.

**Architecture version**:
An incompatible revision of a Ventris architecture, such as Vanilla v1 versus Vanilla v2.

**Batch**:
Synonym for effective batch, but generally prefer "effective batch" for clarity. Never call the device batch "batch".

**Device batch**:
The subset of an effective batch processed by one device in one forward and backward pass.
_Avoid_: Micro-batch

**Effective batch**:
The complete set of training sequences whose gradients contribute to one optimizer update.
_Avoid_: Global batch
