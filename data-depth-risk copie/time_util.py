def time_util(old_t, new_t):
    duration = new_t - old_t
    dur_min = duration/60.0
    print("--- %s seconds ---" % duration)
    print("--- %s min ---" % dur_min)