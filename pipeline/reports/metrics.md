Held-out signers: 16069, 37055, 61333 (14396 clips, never seen in training or model selection).

| Sign set | Signs | Top-1 | Top-5 / Top-3 |
|---|---:|---:|---:|
| All signs | 250 | 72.1% | 88.1% (top-5) |
| Shipped in the demo | 250 | 72.1% | 85.3% (top-3) |

The shipped set is the largest set found whose top-1 on the validation signers is at least 60%; the test signers above did not take part in choosing it.

"Not sure" below 0.29 calibrated confidence: the demo answers 92% of test clips, and those answers are 77.1% correct.

Teach your own (8-way, 3-shot, cosine nearest prototype on test-signer embeddings): 88.7%.

| Most confused (true -> predicted) | Count | Share of true sign |
|---|---:|---:|
| awake -> wake | 38 | 61% |
| lamp -> snack | 28 | 48% |
| finger -> wait | 23 | 43% |
| quiet -> shhh | 21 | 37% |
| sleep -> nap | 21 | 36% |
| pencil -> pen | 19 | 29% |
| cry -> cheek | 18 | 30% |
| mouth -> lips | 18 | 30% |
| brother -> boy | 17 | 30% |
| pen -> pencil | 17 | 27% |
