Instructions for compiling/running your program and changing parameters:

Compiling:
    Make sure Data1.txt and Data2.txt are in the same directory as the decryptga folder.
    ---> in bash, run: javac dycryptga/Main.java decryptga/Evaluation.java

Running:
    in bash, run: java decryptga.Main

    - The program will automatically run all experiments for both crossover types on both data files.
    - It will generate CSV files in the format: results_[dataFile]_[crossoverType]_[experimentName].csv
    --->The CSV contains the generation number, average best fitness and average population across 5 runs.

Changing Parameters:
    All parameteres can be changed by editing decrypt/Main.java.
    - Open the file
    - Navigate to the section near the top of the program that looks like:
        //|*******************************************************************|
        //|---------------------------CONFIGURATION---------------------------|
        //|*******************************************************************|
    - Change parameters such as:
        - Population size
        ---> private static final int defaultPopsize = 200;  //Change this number
        - Maximum Generations
        ---> private static final int defaultMaxGen = 1000;  //Change this number
        - Tournament Size (K)
        ---> private static final int defaultTourn = 3;  //Valid values: 2, 3, 4, 5
        - Elitism Count
        ---> private static final int defaultElite = 1;  //Number of elite individuals
        - Number of runs per configuration
        ---> private static final int runsPerConfig = 5;  //Change this number (min 5)
        - Crossover/mutation rates
        --->private static final ExperimentConfig[] experiments = new ExperimentConfig[] {
                new ExperimentConfig("a_100c_0m", 1.0, 0.0),    //100% crossover, 0% mutation
                new ExperimentConfig("b_100c_10m", 1.0, 0.10),  //100% crossover, 10% mutation
                new ExperimentConfig("c_90c_0m", 0.90, 0.0),    //90% crossover, 0% mutation
                new ExperimentConfig("d_90c_10m", 0.90, 0.10),  //90% crossover, 10% mutation
                new ExperimentConfig("e_90c_1m", 0.90, 0.01)    //90% crossover, 1% mutation
            };

    - After making the changes:
        - Save Main.java
        - Compile
        - Run java decryptga.Main


