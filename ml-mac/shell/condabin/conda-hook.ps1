$Env:CONDA_EXE = "/Users/backupuser/MA/ml-mac/bin/conda"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "/Users/backupuser/MA/ml-mac"
$Env:_CONDA_EXE = "/Users/backupuser/MA/ml-mac/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $False}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs