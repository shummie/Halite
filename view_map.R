library(magrittr)
library(data.table)
library(plotly)
setwd("C:/Users/Shummie/Documents/Python/Halite/replays/Maps")
as.matrix(data.table::fread("maps144.txt")) %>% {plot_ly(z = ., type="surface")}

as.matrix(data.table::fread("distance_from_owned30.txt")) %>% {plot_ly(z = ., type="heatmap")}

rot <- function(x) t(apply(x, 2, rev))
rotate <- function(x) rot(rot(rot(x)))

as.matrix(data.table::fread("distance_from_owned1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_avg1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_max1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_wtd1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("recover_wtd1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("strength_map1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("production_map1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("global_square_map1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
