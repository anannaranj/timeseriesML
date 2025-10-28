from cleaning import cleaning, asFreq

# get cleaned Dataframe
df = cleaning()


# Make it a day frequency

# That is what the machine is going to learn
# as it will take ages to learn it by minutes
df = asFreq(df, "D")

# just a simple commit because I saw gh registering me not as a github user and just as a git user for some reason
