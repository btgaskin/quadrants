# CSR revision evidence supplement

[Revised API smoke measurement](api-smoke/README.md) contains the M5 raw receipt, reproduction script, matched dense/direct-call baselines and measured sources. These results retain their original source identity.

[Portability checks](portability/README.md) contain the CPU/Metal/CUDA and native Linux CPU API receipt, exact commands, T4 logs/JUnit and source identities. The code changes that followed the API smoke measurement concern metadata-assertion ordering; numerical multiplication and typed binding are unchanged.

The maintained PR contains the generic operator, focused tests and concise docs. These evidence artifacts remain separate.
