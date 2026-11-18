# zeroshot_detection_segmentation
A simple pet project to experiment with SOTA zero-shot VLM object-detection and image-segmentation algorithms. 

## Formatting

YAPF is configured in [.style.yapf](./.style.yapf) with a 120-character line limit
and vertically aligned wrapped arguments. With YAPF installed, run
`python -m yapf --recursive --in-place .` from the repository root to format the
Python files, or `python -m yapf --recursive --diff .` to preview changes.

VS Code is configured to format Python files on save using the `eeyore.yapf`
extension.
