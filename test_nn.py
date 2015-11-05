import nn
import numpy as np
import os

def test_init(seed = 16):
    global train_set_complete
    global test_set
    global lbl_test
    train = np.loadtxt('train.csv', delimiter= ',', skiprows= 1, dtype= int)
    np.random.seed(seed)
    l = len(train)
    train_ind = np.random.choice(range(l), replace = False, size = np.floor(l * 0.75))
    train_set_complete = train[train_ind, :]
    test_ind = np.delete(np.array(range(l)), train_ind)
    test_set = train[test_ind, :]
    lbl_test = test_set[:, 0]
    test_set = np.delete(test_set, 0, 1)
    lbl_test[lbl_test == 0] = 10

def write_results(result, test_name):
    wd = os.getcwd()
    try:
        if not os.path.exists("results/"):
            os.makedirs("results/")

        setwd()
        today = Sys.Date()
        dir.create(file.path(test_name))
        setwd(file.path(test_name))
        dir.create(file.path(today))
        setwd(file.path(today))
        name = paste(
            format(Sys.time(), "%H-%M-%S"),
            test_name,
            sep = "_"
        )
        write.csv(x = result, file = paste0(name, ".csv"))
        setwd(wd)
    }, error = function(cond) {
        message("Original error message:")
        message(cond)
    }, finally = {
        setwd(wd)
    })
}

# test for train set size
test_learning_size = function(min_size = 500, max_size = 5000, step_size = 500) {
    
    size = c()
    m_time = c()
    acc = c()
    
    test_init()
    
    # test for hidden layer size
    res = data.frame(training_size = numeric(), accuracy = numeric(), 
                      learning_speed = numeric(), user_time = numeric(),
                      sys_time = numeric(), elapsed_time = numeric())
    for(ind in seq.int(min_size, max_size, step_size)) {
        print(ind)
        hl = 25
        alfa = 1
        init(input_size = 28 * 28, hidden_size = hl, output_size = 10, alfa_ = alfa, seed = 123)
        
        train_set = train_set_complete[1:ind, ]
        
        lbl_train = train_set[, 1]
        train_set[, 1] = NULL
        lbl_train[lbl_train == 0] = 10 
        
        print("Learn")
        ptm = proc.time()
        learning(x = train_set, lbl = lbl_train)
        m_time = proc.time()
        print("Predict")
        predict_res = apply(test_set, 1, predict)
        s = sum(predict_res == lbl_test)
        proc_n = s/length(lbl_test)
        # print(hl)
        print(proc_n)
        new_row = data.frame(training_size = ind, 
                              accuracy = proc_n, 
                              learning_speed = alfa, 
                              hidden_layer_size = hl,
                              user_time = m_time["user.self"],
                              sys_time = m_time["sys.self"], 
                              elapsed_time = m_time["elapsed"])
        res = rbind(res, new_row)
    }
    write_results(res, "size_measurements")
    res
}

# test for hidden layer size
test_hidden_layer_size = function(min_size = 15, max_size = 50, step_size = 5) {
    
    size = c()
    m_time = c()
    acc = c()
    
    test_init()
    
    # test for hidden layer size
    res = data.frame(training_size = numeric(), accuracy = numeric(), 
                      learning_speed = numeric(), user_time = numeric(),
                      sys_time = numeric(), elapsed_time = numeric())
    
    for(hl in seq.int(min_size, max_size, step_size)) {
        print(hl)
        ind = 3500
        alfa = 1
        init(input_size = 28 * 28, hidden_size = hl, output_size = 10, alfa_ = alfa, seed = 123)
        
        train_set = train_set_complete[1:ind, ]
        
        lbl_train = train_set[, 1]
        train_set[, 1] = NULL
        lbl_train[lbl_train == 0] = 10 
        
        print("Learn")
        ptm = proc.time()
        learning(x = train_set, lbl = lbl_train)
        m_time = proc.time()
        print("Predict")
        predict_res = apply(test_set, 1, predict)
        s = sum(predict_res == lbl_test)
        proc_n = s/length(lbl_test)
        # print(hl)
        print(proc_n)
        new_row = data.frame(training_size = ind, 
                              accuracy = proc_n, 
                              learning_speed = alfa, 
                              hidden_layer_size = hl,
                              user_time = m_time["user.self"],
                              sys_time = m_time["sys.self"], 
                              elapsed_time = m_time["elapsed"])
        res = rbind(res, new_row)
    }
    
    write_results(res, "hidden_layer_size_measurements")
    res
}

# test for train speed
test_learning_speed = function(min_speed = 0.1, max_speed = 2, step_size = 0.1) {
    
    size = c()
    m_time = c()
    acc = c()
    
    test_init()
    
    res = data.frame(training_size = numeric(), accuracy = numeric(), 
                      learning_speed = numeric(), user_time = numeric(),
                      sys_time = numeric(), elapsed_time = numeric())
    
    for(alf in seq.int(min_speed, max_speed, step_size)) {
        print(alf)
        ind = 3500
        alfa = alf
        hl = 25
        init(input_size = 28 * 28, hidden_size = hl, output_size = 10, alfa_ = alfa, seed = 123)
        
        train_set = train_set_complete[1:ind, ]
        
        lbl_train = train_set[, 1]
        train_set[, 1] = NULL
        lbl_train[lbl_train == 0] = 10 
        
        print("Learn")
        ptm = proc.time()
        learning(x = train_set, lbl = lbl_train)
        m_time = proc.time() - ptm
        print("Predict")
        predict_res = apply(test_set, 1, predict)
        s = sum(predict_res == lbl_test)
        proc_n = s/length(lbl_test)
        # print(hl)
        print(proc_n)
        new_row = data.frame(training_size = ind, 
                              accuracy = proc_n, 
                              learning_speed = alfa, 
                              hidden_layer_size = hl,
                              user_time = m_time["user.self"],
                              sys_time = m_time["sys.self"], 
                              elapsed_time = m_time["elapsed"])
        res = rbind(res, new_row)
    }
    write_results(res, "training_speed_measurements")
    res
}

test_all = function() {
    print("Test learning size")
    test_learning_size(500, 7000, 500)
    print("Test hidden layer size")
    test_hidden_layer_size(15, 50, 5)
    print("Test learning speed")
    test_learning_speed(0.1, 3, 0.1)
    NULL
}