library(tseries)
library(TSstudio)
library(forecast)
library(xts)
library(tidyverse)

data <- read.csv("./project/.data/bejing_air_quality/cleaned/PRSA_Data_Aotizhongxin_20130301-20170228.csv")


data <- data[, -14]
data <- data[, -12]

head(data)

date <- c("2013-03-01 00:00:00")
#dates <- seq(as.POSIXct(date), as.POSIXct("2017-02-28 23:00:00"), by="hour")

#dates

#data2 <- cbind(date=dates, data)

head(data2)

times <- strptime(data[, "timestamp"], format="%Y-%m-%d %H:%M:%S", tz="EST")
data2 <- xts(data[, -1], order.by = times)
#data <- ts(data, start=dates[1], end=dates[length(dates)])
#data <- ts(as.numeric(data), frequency =35064, start=dates[1], end=dates[length(dates)])

data.tbl <- rownames_to_column(as_tibble(data2, rownames=NA))
head(data.tbl)
head(data2)

pm25 <- data2[,1]
ts_plot(pm25, 
        title = "PM 2.5",
        Ytitle = "Scaled Concentration",
        Xtitle = "Measurement")
# ts_plot(data.tbl[, 1:2], 
#         title = "PM 2.5",
#         Ytitle = "Scaled Concentration",
#         Xtitle = "Measurement")
ts_quantile(pm25, period = "weekdays")

ts_quantile(pm25, period = "monthly")


# Seasonal plot


ts_seasonal(to.monthly(pm25)[,4])


# Heatmap plot
ts_heatmap(pm25)


ts_lags(pm25)
