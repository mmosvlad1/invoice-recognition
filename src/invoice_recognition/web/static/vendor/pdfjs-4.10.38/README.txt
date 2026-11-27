PDF.js 4.10.38 (npm package pdfjs-dist), the PDF renderer of the page's document viewer.

Source:    https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-4.10.38.tgz
Integrity: sha512-/Y3fcFrXEAsMjJXeL9J8+ZG9U01LbuWaYypvDW2ycW1jL269L3js3DVBjDJ0Up9Np1uqDXsDrRihHANhZOlwdQ==
           (the registry's dist.integrity of the tarball, checked before extracting)
Licence:   Apache License 2.0, the package's own LICENSE file, next to this one.

build/pdf.min.mjs and build/pdf.worker.min.mjs of the package, unmodified. Not shipped:
source maps, cmaps/ (CJK fonts without embedded CMaps) and standard_fonts/ (the
browser's fonts stand in for non-embedded standard fonts). Pinned to the version the
page was designed and checked with; no known advisory affects it. tests/test_web_vendor.py
holds the files to the sums below.

sha256                                                            bytes    file
27fc2a057a00f92a4334ad06e17dbd7259912954e9fb7f76400bcca5fd190a9c  352645   pdf.min.mjs
1baa1844c89c80a5b2797c916e75ab29254be46d8e9cb53cb6364d7aad84be36  1375838  pdf.worker.min.mjs
0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594  10174    LICENSE
