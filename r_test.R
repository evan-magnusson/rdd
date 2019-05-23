library("rdd")

data <- read.csv('test_data.csv')

h <- IKbandwidth(data$x, data$y)
h