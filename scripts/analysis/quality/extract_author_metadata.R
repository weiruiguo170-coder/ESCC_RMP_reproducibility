#!/usr/bin/env Rscript

# Minimal, read-only extraction of cell metadata from an author-provided Seurat
# RDS. This intentionally avoids installing or loading the full Seurat package.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: extract_author_metadata.R <input.rds> <output.csv>")
}

input_path <- args[[1L]]
output_path <- args[[2L]]
obj <- readRDS(input_path)

message("class: ", paste(class(obj), collapse = ","))


metadata <- attr(obj, "meta.data", exact = TRUE)
if (is.null(metadata) && is.list(obj) && "meta.data" %in% names(obj)) {
  metadata <- obj[["meta.data"]]
}

if (is.null(metadata) || !is.data.frame(metadata)) {
  stop("The RDS was readable, but no data.frame meta.data component was found")
}

cell_id <- rownames(metadata)
if (is.null(cell_id) || anyNA(cell_id) || any(cell_id == "")) {
  stop("meta.data row names do not provide complete cell identifiers")
}

output <- data.frame(cell_id = cell_id, metadata, check.names = FALSE)
encode <- function(z) ifelse(is.na(z), "NA", paste0('"', gsub('"', '""', as.character(z), fixed=TRUE), '"'))
lines <- c(paste(encode(names(output)), collapse=","), vapply(seq_len(nrow(output)), function(i) paste(vapply(output, function(col) encode(col[i]), character(1)), collapse=","), character(1)))
writeLines(lines, output_path, useBytes=TRUE)
message("metadata rows: ", nrow(output), "; columns: ", ncol(output))

