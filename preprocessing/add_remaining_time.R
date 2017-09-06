#files = c ("train_bpi12.csv", "test_bpi12.csv")
files <- list.files()[grep(paste("^BPIC15(?=.*\\_f2.csv)",sep=''), list.files(), perl=TRUE)]

for (File in files) {
  dat = read.csv(File,sep = ";",check.names = FALSE)
  print(File)
  
  names(dat)[which(names(dat)=="Case ID")]="sequence_nr"
  names(dat)[which(names(dat)=="Complete Timestamp")]="time"
  
  dat = dat[order(dat$sequence_nr),]
  
  sequence_length_dt <- as.data.frame(table(dat$sequence_nr))
  colnames(sequence_length_dt) <- c('sequence_nr', 'seq_length')
  sequence_length_dt$sequence_nr = as.numeric(as.character(sequence_length_dt$sequence_nr))
  sequence_length_dt = sequence_length_dt[order(sequence_length_dt$sequence_nr),]
  
  print(dat$time[2])
  dat$time = strptime(dat$time,"%Y-%m-%d %H:%M:%S")
  dat$remtime = -1
  
  k = 0
  for(i in 1:nrow(sequence_length_dt)){
    #curtrace = dat$Case.ID[i]
    if(i %% 100 == 0) cat(i, "out of", nrow(sequence_length_dt), "processed \n")
    for(j in 1:sequence_length_dt$seq_length[i]) {
      #time_trace_finish = dat$time[k+sequence_length_dt$seq_length[i]]
      time_trace_finish = max(dat$time[dat$sequence_nr==sequence_length_dt$sequence_nr[i]])
      dat$remtime[k+j] = difftime(time_trace_finish,dat$time[k+j],units = "secs")
    }
    k = k+sequence_length_dt$seq_length[i]
    
  }
  print(dim(dat))
  names(dat)[which(names(dat)=="sequence_nr")]="Case ID"
  names(dat)[which(names(dat)=="time")]="Complete Timestamp"
  write.table(dat,file = sprintf("%s_time",File),sep = ";",quote = F,row.names = F)
  
}

