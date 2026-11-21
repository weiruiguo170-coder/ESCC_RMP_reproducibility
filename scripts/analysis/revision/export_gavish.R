# Export the official author object without altering memberships.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Usage: export_gavish.R <author.RDS> <memberships.tsv>')
x <- readRDS(args[[1L]])
z <- do.call(rbind, lapply(names(x), function(ct) {
  do.call(rbind, lapply(names(x[[ct]]), function(mp) {
    data.frame(cell_type=ct, reference_program=mp,
      rank=seq_along(x[[ct]][[mp]]), gene=x[[ct]][[mp]])
  }))
}))
write.table(z, args[[2L]], sep='\t', quote=FALSE, row.names=FALSE)
