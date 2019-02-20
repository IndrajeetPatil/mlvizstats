# Book: Introduction to Machine Learning with R:Rigorous Mathematical Analysis
# Author: Scott V. Burger

#======================================= Chapter 2 ===========================

library(tidyverse)
library(magrittr)
library(ggplot2)

# global datasets to use
# for reproducibility
set.seed(123)

# creating indices to choose rows from the data
train_indices <-
  base::sample(x = base::seq_len(length.out = nrow(mtcars)),
               size = floor(0.8 * nrow(mtcars)))

# training dataset
train <- mtcars[train_indices,]

# testing dataset
test <- mtcars[-train_indices,]


#================================= regression ================================

# page no. 23-24
lm_rmse <- function(data, x, y, split_size = 0.8) {
  # creating a dataframe
  data <-
    dplyr::select(
      .data = data,
      x = !!rlang::enquo(x),
      y = !!rlang::enquo(y)
    )

  # for reproducibility
  set.seed(123)

  # creating indices to choose rows from the data
  train_indices <-
    base::sample(x = base::seq_len(length.out = nrow(data)),
                 size = floor(split_size * nrow(data)))

  # training dataset
  train <- data[train_indices,]

  # testing dataset
  test <- data[-train_indices,]

  # running a model on the training dataset
  model_train <-
    stats::lm(
      formula = stats::as.formula(y ~ x),
      data = train,
      na.action = na.omit
    )

  # predicted response from model on the testing dataset
  test$output <-
    stats::predict(object = model_train, data.frame(x = test$x))

  # return RMSE value by comparing predicted versus observed values
  return(sqrt(sum(test$y - test$output) ^ 2 / nrow(test)))
}

# example (reproduces result from the book)
lm_rmse(data = mtcars, x = disp, y = mpg)


logit_boost <- function(data, x, y, split_size = 0.8) {
  # creating a dataframe
  data <-
    dplyr::select(.data = data,
                  !!rlang::enquo(x),
                  !!rlang::enquo(y))

  # for reproducibility
  set.seed(123)

  # creating indices to choose rows from the data
  train_indices <-
    base::sample(x = base::seq_len(length.out = nrow(data)),
                 size = floor(split_size * nrow(data)))

  # training dataset
  train <- data[train_indices,]

  # testing dataset
  test <- data[-train_indices,]

  # defining label column we are interested in and everything else
  label_train <-
    train %>% dplyr::select(.data = ., !!rlang::enquo(x)) %>% as.vector(x = .)

  data_train <-
    train %>% dplyr::select(.data = ., -!!rlang::enquo(x))

  # training model (y ~ x)
  logit_model <-
    caTools::LogitBoost(xlearn = data_train,
                        ylearn = dplyr::pull(.data = label_train, var = -1))

  # prediction
  predicted_label <-
    predict(object = logit_model, test, type = "raw")
}

#logit_boost(data = mtcars, x = am, y = mpg)

#========================= classification  =============================

# custom function to get both sum of squares and cluster centres simultaneously
#' @inheritParams stats::kmeans

kmeans_broom <- function(x,
                         centers,
                         iter.max = 10,
                         nstart = 1,
                         algorithm = "Hartigan-Wong",
                         trace = FALSE) {
  # removing non-numeric variables from the dataframe
  x %<>%
    dplyr::select_if(.tbl = ., .predicate = purrr::is_bare_numeric)

  # sum of squares
  kmeans_glance <-
    function(x = x,
             centers = centers,
             iter.max = iter.max,
             nstart = nstart,
             algorithm = algorithm,
             trace = trace) {
      broom::glance(
        x = stats::kmeans(
          x = x,
          centers = centers,
          iter.max = iter.max,
          nstart = nstart,
          algorithm = algorithm,
          trace = trace
        )
      ) %>%
        tibble::as_tibble(x = .)
    }

  # cluster centres
  kmeans_tidy <-
    function(x = x,
             centers = centers,
             iter.max = iter.max,
             nstart = nstart,
             algorithm = algorithm,
             trace = trace) {
      broom::tidy(
        x = stats::kmeans(
          x = x,
          centers = centers,
          iter.max = iter.max,
          nstart = nstart,
          algorithm = algorithm,
          trace = trace
        )
      ) %>%
        tibble::as_tibble(x = .)
    }

  # cluster centres
  kmeans_augment <-
    function(x = x,
             centers = centers,
             iter.max = iter.max,
             nstart = nstart,
             algorithm = algorithm,
             trace = trace) {
      broom::augment(
        x = stats::kmeans(
          x = x,
          centers = centers,
          iter.max = iter.max,
          nstart = nstart,
          algorithm = algorithm,
          trace = trace
        ),
        x
      ) %>%
        tibble::as_tibble(x = .)
    }

  # running both functions simultaneously
  purrr::invoke_map(
    .x = list(
      list(
        x = x,
        centers = centers,
        iter.max = iter.max,
        nstart = nstart,
        algorithm = algorithm,
        trace = trace
      )
    ),
    .f = c("kmeans_glance", "kmeans_tidy", "kmeans_augment")
  )

}

# plotting the clusters
kmeans_clusters_plot <-
  function(data,
           x,
           y,
           cluster.center.labels = "none",
           geom.text = FALSE,
           geom.text.label = NULL,
           xlab = NULL,
           ylab = NULL,
           caption = NULL,
           title = NULL) {
    # creating caption containing information
    subtitle.text <- na.omit(data$caption)

      # importing data
      data <-
        dplyr::select(
          .data = data,
          x = !!rlang::enquo(x),
          y = !!rlang::enquo(y),
          dplyr::everything()
        )

    # creating the basic plot
    plot <- ggplot2::ggplot(data = data,
                            mapping = ggplot2::aes(x = x, y = y),
                            na.rm = TRUE) +
      ggplot2::geom_jitter(
        data = dplyr::filter(data, !is.na(.cluster)),
        ggplot2::aes(
          x = x,
          y = y,
          color = .cluster,
          shape = .cluster
        ),
        size = 3,
        na.rm = TRUE,
        inherit.aes = FALSE,
        alpha = 0.7
      ) +
      ggplot2::geom_point(
        data = data,
        ggplot2::aes(x = x1, y = x2),
        size = 8,
        shape = "x",
        inherit.aes = FALSE,
        na.rm = TRUE
      )

    # adding labels to the plot about the cluster statistics, like coordinates, size, etc.
    if (cluster.center.labels != "none") {
      if (cluster.center.labels == "coordinates") {
        plot <- plot +
          ggrepel::geom_label_repel(
            data = dplyr::filter(data, !is.na(x1)),
            mapping = ggplot2::aes(x = x1, y = x2, label = label.coordinates),
            fontface = "bold",
            inherit.aes = FALSE,
            direction = "both",
            na.rm = TRUE
          )
      } else if (cluster.center.labels == "size") {
        plot <- plot +
          ggrepel::geom_label_repel(
            data = dplyr::filter(data, !is.na(x1)),
            mapping = ggplot2::aes(x = x1, y = x2, label = label.size),
            fontface = "bold",
            inherit.aes = FALSE,
            direction = "both",
            na.rm = TRUE,
            parse = TRUE
          )
      } else if (cluster.center.labels == "variance") {
        plot <- plot +
          ggrepel::geom_label_repel(
            data = dplyr::filter(data, !is.na(x1)),
            mapping = ggplot2::aes(x = x1, y = x2, label = label.variance),
            fontface = "bold",
            inherit.aes = FALSE,
            direction = "both",
            na.rm = TRUE
          )
      }
    }

    plot <- plot +
      ggplot2::labs(
        x = xlab,
        y = ylab,
        title = title,
        caption = caption,
        subtitle = subtitle.text
      ) +
      ggstatsplot::theme_mprl(ggtheme = theme_grey())

    return(plot)
  }

tidy_pca <- function(data, output = "pca_tidy") {
  # importing the data
  df <- data

  # running PCA with tidyverse packages
  df_pca <- df %>%
    tidyr::nest(data = ., .key = "data") %>%
    dplyr::mutate(
      .data = .,
      pca = purrr::map(
        .x = data,
        .f = ~ stats::prcomp(
          x = .x %>% dplyr::select_if(.tbl = ., .predicate = purrr::is_bare_numeric),
          center = TRUE,
          scale = TRUE
        )
      ),
      pca_tidy = purrr::map2(
        .x = pca,
        .y = data,
        .f = ~ broom::augment(x = .x, data = .y)
      )
    )

  # explained variance
  pca_exp_var <- df_pca %>%
    tidyr::unnest(data = ., pca_tidy) %>%
    dplyr::summarize_at(
      .tbl = .,
      .vars = dplyr::vars(dplyr::contains("PC")),
      .funs = dplyr::funs(var)
    ) %>%
    tidyr::gather(data = .,
                  key = pc,
                  value = variance) %>%
    dplyr::mutate(
      .data = .,
      var_exp = paste0(
        ggstatsplot::specify_decimal_p(x = variance / sum(variance) * 100, k = 2),
        " %",
        sep = ""
      )
    ) %>%
    tibble::rownames_to_column(., var = "rowid") %>%
    tibble::as_tibble(x = .) %>%
    purrrlyr::by_row(
      .d = .,
      ..f = ~ paste("PC",
                    .$rowid,
                    " (",
                    .$var_exp,
                    ")",
                    sep = ""),
      .collate = "rows",
      .to = "pc_label",
      .labels = TRUE
    ) %>%
    dplyr::select(.data = ., -rowid)

  if (output == "pca_tidy") {
    pca_tidy_df <- df_pca %>%
      tidyr::unnest(data = ., pca_tidy) %>%
      dplyr::rename(.data = ., PC1 = .fittedPC1, PC2 = .fittedPC2, PC3 = .fittedPC3, PC4 = .fittedPC4)

    return(pca_tidy_df)
  } else if (output == "pca_labels") {
    return(pca_exp_var)
  }

}


#' @inheritParams stats::kmeans
#' @inheritDotParams combine_plots

ggkmeansstats <- function(data,
                          xlab = NULL,
                          ylab = NULL,
                          caption = NULL,
                          title = NULL,
                          cluster.center.labels = "none",
                          scale = TRUE,
                          centers,
                          iter.max = 10,
                          nstart = 1,
                          algorithm = "Hartigan-Wong",
                          trace = FALSE,
                          output = "plot",
                          ...) {

  if (dim(dplyr::select_if(.tbl = data, .predicate = purrr::is_bare_numeric))[[2]] == 2L) {
    # preparing labels from given dataframe
    lab.df <-
      colnames(dplyr::select_if(.tbl = data, .predicate = purrr::is_bare_numeric))
    # if xlab is not provided, use the variable x name
    if (is.null(xlab)) {
      xlab <- lab.df[1]
    }
    # this label with be later used while creating plots
    x_var_name <- lab.df[1]
    # if ylab is not provided, use the variable y name
    if (is.null(ylab)) {
      ylab <- lab.df[2]
    }
    # this label with be later used while creating plots
    y_var_name <- lab.df[2]

  } else {
    # if xlab is not provided, use the variable for first principal component
    if (is.null(xlab)) {
      xlab <- tidy_pca(data = data, output = "pca_labels")$pc_label[[1]]
    }
    # if ylab is not provided, use the variable for second principal component
    if (is.null(ylab)) {
      ylab <- tidy_pca(data = data, output = "pca_labels")$pc_label[[2]]
    }
  }

  # dataframe on which kmeans is to be carried out
  kmeans_df <- data %>%
    tibble::as_tibble(x = .)

  # standardizing the numeric variables in the dataframe
  if (isTRUE(scale)) {
    kmeans_df %<>%
      purrr::map_if(
        .x = .,
        .p = purrr::is_bare_numeric,
        .f = ~ base::scale(
          x = .,
          center = TRUE,
          scale = TRUE
        )
      ) %>%
      dplyr::bind_rows()
  }

  if (dim(dplyr::select_if(.tbl = data, .predicate = purrr::is_bare_numeric))[[2]] == 2L) {
  # creating a combined dataframe
  result_df <- purrr::pmap(
    .l = list(
      x = list(kmeans_df),
      centers = centers,
      iter.max = list(iter.max),
      nstart = list(nstart),
      algorithm = list(algorithm),
      trace = list(trace)
    ),
    .f = kmeans_broom
  )
  } else {
    kmeans_pca_df <- tidy_pca(data = kmeans_df, output = "pca_tidy")
    # creating a combined dataframe
    result_df <- purrr::pmap(
      .l = list(
        x = list(dplyr::select(.data = kmeans_pca_df, PC1:PC4)),
        centers = centers,
        iter.max = list(iter.max),
        nstart = list(nstart),
        algorithm = list(algorithm),
        trace = list(trace)
      ),
      .f = kmeans_broom
    )
}
  # creating a combined dataframe with results from all k values
  options(warn = -1)
  result_df %<>%
    purrr::map_dfr(.x = .,
                   .f = dplyr::bind_rows,
                   .id = "index") %>%
    dplyr::mutate_at(.tbl = .,
                     .vars = "index",
                     .funs = ~ as.factor(.))

  # joing PCA results with the results from kmeans analysis
  # result_df %<>%
  #   dplyr::full_join(x = ., y = tidy_pca(kmeans_df, output = "pca_tidy"))

  # this is basically to mute warnings
  # Warning in bind_rows_(x, .id) :
  # Unequal factor levels: coercing to character
  options(warn = 0)

  if (length(levels(result_df$index)) != 1) {
    result_df %<>%
      dplyr::group_by(.data = ., index) %>%
      dplyr::mutate(.data = ., k = max(cluster, na.rm = TRUE)) %>%
      dplyr::ungroup(x = .)
  } else {
    result_df %<>% dplyr::mutate(.data = ., k = "1")
  }

  # the center coordinates will also have to reduced in dimensionality
  # if (dim(dplyr::select_if(.tbl = data, .predicate = purrr::is_bare_numeric))[[2]] != 2L) {
  #
  #   centers.coord.pca <-
  #     dplyr::select(.data = result_df, dplyr::matches(match = "^x[0-9]$"), k, cluster) %>% dplyr::filter(.data = ., !is.na(x1)) %>%
  #     tidy_pca(data = ., output = "pca_tidy") %>%
  #     dplyr::select(.data = ., -c(dplyr::matches(match = "^x[0-9]$"), .rownames)) %>%
  #     dplyr::rename(.data = ., x1 = PC1, x2 = PC2) %>%
  #     dplyr::select(.data = ., -c(dplyr::matches(match = "^PC[0-9]$")))
  #
  #   result_df %<>%
  #     dplyr::select(.data = ., -c(dplyr::matches(match = "^x[0-9]$")))
  #
  #   result_df <-
  #     dplyr::left_join(x = result_df,
  #                      y = centers.coord.pca,
  #                      by = c("k", "cluster"))
  # }

  #================================================ creating labels and other text elements ===================================================

  result_df %<>%
    purrrlyr::by_row(
      .d = .,
      ..f = ~ paste0(
        "Sum of Squares (% of total): within = ",
        ggstatsplot::specify_decimal_p(x = (.$tot.withinss / .$totss) *
                                         100, k = 2),
        "%, ",
        "between = ",
        ggstatsplot::specify_decimal_p(x = (.$betweenss / .$totss) * 100, k = 2),
        "%"
      ),
      .collate = "rows",
      .to = "caption",
      .labels = TRUE
    ) %>%
    dplyr::mutate(
      .data = .,
      caption = dplyr::case_when(!is.na(totss) ~ caption,
                                 is.na(totss) ~ NA_character_)
    ) %>%
    purrrlyr::by_row(
      .d = .,
      ..f = ~ paste0(
        "(",
        ggstatsplot::specify_decimal_p(x = .$x1, k = 3),
        ",",
        ggstatsplot::specify_decimal_p(x = .$x2, k = 3),
        ")"
      ),
      .collate = "rows",
      .to = "label.coordinates",
      .labels = TRUE
    ) %>%
    purrrlyr::by_row(
      .d = .,
      ..f = ~ paste("list(italic(n)==",
                    .$size,
                    ")",
                    sep = ""),
      .collate = "rows",
      .to = "label.size",
      .labels = TRUE
    ) %>%
    dplyr::group_by(.data = ., index, k) %>%
    dplyr::mutate(.data = ., total.ss = sum(totss, na.rm = TRUE)) %>%
    dplyr::ungroup(x = .) %>%
    purrrlyr::by_row(
      .d = .,
      ..f = ~ paste0(
        "SS: ",
        ggstatsplot::specify_decimal_p(x = (.$withinss / .$total.ss) *
                                         100, k = 2),
        "%",
        sep = ""
      ),
      .collate = "rows",
      .to = "label.variance",
      .labels = TRUE
    ) %>%
    dplyr::mutate(
      .data = .,
      label.coordinates = dplyr::case_when(!is.na(x1) ~ label.coordinates,
                                           is.na(x1) ~ NA_character_)
    ) %>%
    dplyr::mutate(
      .data = .,
      label.size = dplyr::case_when(!is.na(x1) ~ label.size,
                                    is.na(x1) ~ NA_character_)
    ) %>%
    dplyr::mutate(
      .data = .,
      label.variance = dplyr::case_when(!is.na(x1) ~ label.variance,
                                        is.na(x1) ~ NA_character_)
    ) %>%
    dplyr::mutate(
      .data = .,
      title.text = dplyr::case_when(
        !is.na(k) ~ paste("No. of clusters: ", .$k, sep = ""),
        is.na(k) ~ NA_character_
      )
    ) %>%
    dplyr::select(.data = ., -total.ss)

  #============================= plot =======================================

  if (dim(dplyr::select_if(.tbl = data, .predicate = purrr::is_bare_numeric))[[2]] == 2L) {
    if (output == "plot") {
      if (length(levels(result_df$index)) == 1) {
        plot <- kmeans_clusters_plot(
          data = result_df,
          x = x_var_name,
          y = y_var_name,
          cluster.center.labels = cluster.center.labels,
          xlab = xlab,
          ylab = ylab,
          caption = caption,
          title = title
        )
      } else {
        result_df_list <-
          result_df %>% split(x = .,
                              f = .$k,
                              drop = TRUE)

        plotlist <- purrr::map(
          .x = result_df_list,
          .f = ~ kmeans_clusters_plot(
            data = .,
            x = x_var_name,
            y = y_var_name,
            cluster.center.labels = cluster.center.labels,
            xlab = xlab,
            ylab = ylab,
            caption = caption,
            title = .$title.text
          )
        )

        plot <- ggstatsplot::combine_plots(plotlist = plotlist, ...)
      }
      return(plot)
    } else if (output == "results") {
      return(result_df)
    }
  } else {
    if (output == "plot") {
      if (length(levels(result_df$index)) == 1) {
        plot <- kmeans_clusters_plot(
          data = result_df,
          x = PC1,
          y = PC2,
          cluster.center.labels = cluster.center.labels,
          xlab = xlab,
          ylab = ylab,
          caption = caption,
          title = title
        )
      } else {
        result_df_list <-
          result_df %>% split(x = .,
                              f = .$k,
                              drop = TRUE)

        plotlist <- purrr::map(
          .x = result_df_list,
          .f = ~ kmeans_clusters_plot(
            data = .,
            x = PC1,
            y = PC2,
            cluster.center.labels = cluster.center.labels,
            xlab = xlab,
            ylab = ylab,
            caption = caption,
            title = .$title.text
          )
        )

        plot <-
          ggstatsplot::combine_plots(plotlist = plotlist, ...)
      }
      return(plot)
    } else if (output == "results") {
      return(result_df)
    }
  }

}
