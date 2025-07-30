# Print header only once
Write-Host "Mode`tProcs`tTime (s)`tIntegral"
Write-Host "----`t-----`t--------`t--------"

# Loop through process counts from 1 to 25
foreach ($n in 1..25) {
    # Capture output of the MPI run
    $output = & mpiexec -n $n ./heavy_mpi_integral.exe

    # Extract only the MPI result line (skip headers from inside the program)
    $result_line = $output | Where-Object { $_ -match "^MPI\s+\d+" }

    # Print cleaned result
    if ($result_line) {
        Write-Host $result_line
    }
}
