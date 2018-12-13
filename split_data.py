import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('data', output="output", seed=1337, ratio=(.8, .0, .2)) # default values
