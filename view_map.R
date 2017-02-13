library(magrittr)
library(data.table)
library(plotly)
library(ggplot2)

setwd("C:/Users/Shummie/Documents/Python/Halite/replays/Maps")

rot <- function(x) t(apply(x, 2, rev))
rotate <- function(x) rot(rot(rot(x)))

as.matrix(data.table::fread("maps144.txt")) %>% {plot_ly(z = ., type="surface")}
as.matrix(data.table::fread("distance_from_owned30.txt")) %>% {plot_ly(z = ., type="heatmap")}

as.matrix(data.table::fread("distance_from_owned1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_avg1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_max1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_wtd1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("recover_wtd1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("strength_map1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("production_map1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("global_square_map16.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("pos_area1.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("pos_area_516.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("prod_over_str_0_2.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}

data.table::fread("value_prod_map_94.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}

map_at_turn_x<- function(x) {
  #data.table::fread(paste0("value_map", x ,".txt")) %>%  apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
  a <- data.table::fread(paste0("value_map_", x ,".txt")) %>%  apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate()
  export(plot_ly(a, z = ., type="heatmap"), tempfile(fileext = ".png"))
  
}

map_at_turn_x(39)



as.matrix(data.table::fread("buildup_multiplier_2.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("value_map_80.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("value_prod_201.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
as.matrix(data.table::fread("buildup_186.txt")) %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("c1_201.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("f_n_201.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("enemy_str_1_154.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("value_prod_183.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("distance_from_combat_zone201.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}
data.table::fread("combat_zone_184.txt") %>% apply(2, function(x) {x[x == 9999] <- NA; x}) %>%  as.matrix() %>% rotate() %>% {plot_ly(z = ., type="heatmap")}

