:: ghc-options taken from https://github.com/jgm/pandoc/blob/2.19.2/.github/workflows/release-candidate.yml#L37
stack build -j4 +RTS -A256m -RTS --copy-bins pandoc
MKDIR %PREFIX%\bin
MOVE %APPDATA%\local\bin\pandoc.exe %PREFIX%\bin\pandoc.exe
