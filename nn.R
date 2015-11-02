input_layer_size <- 28 * 28
hidden_layer_size <- 15
output_layer_size <- 10
alfa <- 1
MAX_INPUT <- 256

theta1 <- matrix(runif(hidden_layer_size * (input_layer_size + 1), min = -0.1, max = 0.1),
                          nrow = hidden_layer_size, ncol = input_layer_size + 1)
theta2 <- matrix(runif(output_layer_size * (hidden_layer_size + 1), min = -0.1, max = 0.1),
                 nrow = output_layer_size, ncol = hidden_layer_size + 1)

init <- function(input_size = 28 * 28, hidden_size = 15, output_size = 10, alfa_ = 1, seed = 16) {
    input_layer_size <<- input_size
    hidden_layer_size <<- hidden_size
    output_layer_size <<- output_size
    alfa <<- alfa_
    set.seed(seed)
    theta1 <- matrix(runif(hidden_layer_size * (input_layer_size + 1), min = -0.1, max = 0.1),
                     nrow = hidden_layer_size, ncol = input_layer_size + 1)
    theta2 <- matrix(runif(output_layer_size * (hidden_layer_size + 1), min = -0.1, max = 0.1),
                     nrow = output_layer_size, ncol = hidden_layer_size + 1)
}

sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

d_sigmoid <- function(x) {
    sigmoid(x) * (1 - sigmoid(x))
}

forward <- function(x) {
    z_in <- theta1 %*% as.numeric(x)
    z <- sigmoid(z_in)
    z <- c(1, z)
    #forward, hiddent to output
    y_in <- theta2 %*% z
    y <- sigmoid(y_in)
    s <- sum(y)
    y <- lapply(y, '/', s)
    list(y, y_in, z, z_in)
}

learning <- function(x_in, lbl_in) {
    for(i in 1:nrow(x_in)) {
        x <- x_in[i, ] / MAX_INPUT
        lbl <- lbl_in[i]
        t <- rep(0, output_layer_size)
        if (lbl == 0) lbl <- 10
        t[lbl] <- 1
        x <- c(1, x)
        #forward, input to hidden
        res <- forward(x)
        y <- as.numeric(res[[1]])
        y_in <- res[[2]]
        z <- res[[3]]
        z_in <- res[[4]]
        
        #back, compute error for output
        sigma2 <- (t - y) * d_sigmoid(y_in)
        d_theta2 <- sapply(z, '*', sigma2) * alfa
        
        sigma_in <- t(theta2) %*% sigma2
        sigma1 <- sigma_in[-1] * d_sigmoid(z_in)
        d_theta1 <- sapply(x, '*', sigma1) * alfa
        theta1 <<- theta1 + d_theta1
        theta2 <<- theta2 + d_theta2
    }
}

predict <- function(x_in) {
    x <- c(1, x_in)
    x <- x / MAX_INPUT
    res <- forward(x)
    which.max(as.numeric(res[[1]]))
}