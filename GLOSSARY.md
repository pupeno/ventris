# Ventris

Ventris is an educational language model project designed to be understood end to end.

## Language

**Architecture**:
A Ventris model design, such as Vanilla, RoPE, or MLA. Architecture distinguishes how a model processes tokens from how it was trained.
_Avoid_: Variant when referring to a model design

**Version**:
A generation of a Ventris architecture, distinguished by backward-incompatible changes to that architecture.

**Variant**:
A training distinction within a Ventris architecture and version, such as Base or Instruct.
_Avoid_: Architecture when referring to a training distinction

**Training run**:
A training history with its own identity, which can span interruptions and resumptions and produce multiple checkpoints.

**Checkpoint**:
A saved snapshot of a model and its training state at a particular training step, sufficient to evaluate the model or resume training.
_Avoid_: Version when referring to a particular set of saved weights

**Resume**:
Continue an interrupted training run from its latest checkpoint, preserving the run's identity.

**Fork**:
Start a new training run from a chosen checkpoint to conduct an experiment, preserving the source run's history.

**Effective batch**:
The complete set of training sequences whose gradients contribute to one optimizer update.
_Avoid_: Global batch

**Device batch**:
The subset of an effective batch processed by one device in one forward and backward pass.
_Avoid_: Micro-batch

**Batch**:
Synonym for effective batch, but generally prefer "effective batch" for clarity. Never call the device batch "batch".
