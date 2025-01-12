################################################################################
# This script requires the following data file:
# "Correlation_results.xlsx"; 
# Shangzhi Lu, Zhicheng Lin , Jan 5, 2025
################################################################################

#### 1. Preparation_____________________________________________________________
## 1.1 Require the following package
package_list <- c("tidyverse","svglite","readxl","openxlsx","lemon","colorspace") # lemon: https://cran.r-project.org/web/packages/lemon/vignettes/capped-axes.html
# new_packages <- package_list[!(package_list %in% installed.packages()[,"Package"])] # can be slow if thousands of packages are installed 
# if(length(new_packages)) install.packages(new_packages)

## 1.2 Load the above package
lapply(package_list, require, character.only = TRUE) # Load multiple packages

## 1.3 Set the working directory to the folder of the current file (note: run within this file rather than on the Console)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#### 2. Import and extract data_________________________________________________
## 2.1 Import data
Plot_data <- read_excel("Correlation_results.xls", na = "---")
colnames(Plot_data)[1] <- "Scene"
Plot_data_long <- Plot_data %>%
  pivot_longer(
    cols = -Scene,        
    names_to = "maps",         
    values_to = "Correlation" 
  ) # Transfer data format from Wide format to Long format

#### 3. Plot line chart_________________________________________________________
## 3.0 Set global theme (to be used for subsequent plotting)
# https://stackoverflow.com/questions/50741128/set-a-theme-and-palette-for-all-plots
Plot_data_long$Scene <- as.factor(Plot_data_long$Scene)

gglayer_theme <- list(
  theme_classic(base_size = 12), # customize the non-data components (titles, labels, fonts, background, gridlines, and legends)
  theme(text = element_text(size = 12, family = "sans"),
        axis.ticks.length = unit(5, "points"), # change tick length on x/y axis
        axis.ticks = element_line(colour = "black", size = 0.25), # to remove x ticks: axis.ticks.x = element_blank()
        axis.text = element_text(size = 12, colour = "black"), # change axis color (default is gray)
        axis.line = element_line(colour = 'black', size = 0.25),
        axis.title = element_text(size = 12),
        plot.title = element_text(vjust = 0.5, hjust = 0.5, size = 12, face='bold'),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 12)),
  coord_capped_cart(bottom = capped_horizontal(), 
                    left = capped_vertical(capped = "both")) # from the "lemon" package: to truncate axis lines
)

P1 <- ggplot(data = Plot_data_long, aes(x = Scene, y = Correlation, fill = maps, colour = maps, shape = maps, group = maps))+
  labs(title="Correlation with attention map", x="Scene ID",y="Pearson correlation coefficient")+  
  geom_point(size = 2.5, alpha = 0.7) + # symbols for data points
  geom_line(aes(linetype = maps), alpha = 0.7) +  # Add alpha here
  geom_segment(aes(x = 1, y = 0, xend = 14, yend = 0), linetype = 3, color = "gray90") +  # Add the customized dashed line segment
  geom_text(aes(x = 1, y = 0.08, label = "U-Net time:\n0.16 ± 0.02 s", hjust = 0), color = "#d95f02", size =12 * 0.351375, family = "sans") +  # U-Net annotation
  geom_text(aes(x = 1, y = 0.24, label = "LLaVA time:\n349.98 ± 0.02 s", hjust = 0), color = "#006096", size =12 * 0.351375, family = "sans") +  # Finetuned LLaVa annotation
  scale_linetype_manual("maps", values = c("dashed", "solid", "solid", "solid"), # Define linetypes
                        breaks = c("human_meaning_maps", "llava_finetune_maps", "llava_raw_maps", "Unet_predictions"),
                        labels = c("Human", "Finetuned LLaVA", "Raw LLaVA", "U-Net")) +
  scale_colour_manual("maps", values = c("#707070","#006096","#4DBBD5","#d95f02"), 
                      breaks = c("human_meaning_maps", "llava_finetune_maps", "llava_raw_maps", "Unet_predictions"),
                      labels = c("Human", "Finetuned LLaVA", "Raw LLaVA", "U-Net")) +
  scale_fill_manual("maps", values = c(NA, NA, NA, NA), # No fill for points (optional)
                    breaks = c("human_meaning_maps", "llava_finetune_maps", "llava_raw_maps", "Unet_predictions"),
                    labels = c("Human", "Finetuned LLaVA", "Raw LLaVA", "U-Net")) +
  scale_shape_manual("maps", values = c(4, 0, 5, 17), 
                     breaks = c("human_meaning_maps", "llava_finetune_maps", "llava_raw_maps", "Unet_predictions"),
                     labels = c("Human", "Finetuned LLaVA", "Raw LLaVA", "U-Net")) +
  scale_x_discrete(breaks = unique(Plot_data_long$Scene),
                   labels = c(048, 049, 054, 059, 061, 064, 067, 068, 073, 074, 075, 076, 079, 080))+  
  scale_y_continuous(breaks = c(-0.3, 0.0, 0.3, 0.6, 0.9), 
                     labels = c(-0.3, 0.0, 0.3, 0.6, 0.9), limits = c(-0.3, 0.9))+
  gglayer_theme+
  theme(axis.text.x = element_text(vjust=0.5, hjust = 0.5), 
        legend.title = element_blank(), 
        legend.position = c(0.50, 0.95),
        legend.key.height = unit(1, "lines"),  # Adjust this value (e.g., 1.5, 2, 2.5)
        legend.background = element_rect(fill = "transparent", colour = NA))+
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE), # Force 1 row for the fill legend
    color = guide_legend(nrow = 1, byrow = TRUE), # Force 1 row for the color legend
    shape = guide_legend(nrow = 1, byrow = TRUE), # Force 1 row for the shape legend
    linetype = guide_legend(nrow = 1, byrow = TRUE) # Force 1 row for the linetype legend
  )

P1

ggsave("Fig1.pdf", plot = P1, device = cairo_pdf, width = 7, height = 4.8)
ggsave("Fig1.svg", plot = P1, device = "svg", width = 7, height = 4.8, units = "in")