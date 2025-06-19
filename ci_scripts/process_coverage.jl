using Coverage

outputfilename = ARGS[1]
folders = ARGS[2:end]

coverage = Vector{FileCoverage}()
for folder in folders
    append!(coverage, process_folder(folder))
end
LCOV.writefile(outputfilename, coverage)

covered_lines, total_lines = get_summary(coverage)
println("Covered lines: ", covered_lines)
println("  Total lines: ", total_lines)
