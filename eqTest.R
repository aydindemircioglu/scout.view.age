library(TOSTER)
# AMR is the best model
bestModel = "AMR"

tbl = read.table(paste("./results/", bestModel, ".test.csv", sep =''), sep = ",", header = T)
tbl["serror"] = tbl["Age"] - tbl["preds"]
tbl["error"] = abs(tbl["Age"] - tbl["preds"])
errs = c(abs(tbl["Age"] - tbl["preds"]))$Age
message("\tMean: ", mean(errs, na.rm = T))
message("\tSD: ", sd(errs, na.rm = T))
over = sum(abs(tbl["Age"] - tbl["preds"]) < 2)
message("\tBeyond 2.0: ", round(100-100*over/dim(tbl)[1],2), "% -- ", dim(tbl)[1]- over, " / ", dim(tbl)[1] )
over = sum(abs(tbl["Age"] - tbl["preds"]) < 4)
message("\tBeyond 4.0: ", round(100-100*over/dim(tbl)[1],2), "% -- ", dim(tbl)[1]- over, " / ", dim(tbl)[1] )
message("\tSigned error:", mean(tbl$serror), na.rm = T)
message("\tSigned mean:", sd(tbl$serror), na.rm = T)

test = dataTOSTone(data =tbl, vars="error", mu = 0, low_eqbound = 0, high_eqbound = 2.0, eqbound_type = "raw",  desc  = T, plots = T)
print(test)

print ("Correlation:", cor(tbl$Age, tbl$preds))

#
