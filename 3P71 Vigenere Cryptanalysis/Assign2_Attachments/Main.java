package decryptga;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main{

    //|*******************************************************************|
    //|---------------------------CONFIGURATION---------------------------|
    //|*******************************************************************|

    //Files to run experiments on
    private static final String[] dataFiles={"Data1.txt", "Data2.txt"};

    private static final int defaultPopsize=200;//Population size
    private static final int defaultMaxGen=1000; //Max generations
    private static final int defaultTourn=3; //Tournament K (2-5 allowed)
    private static final int defaultElite=1;//Number of elite individuals
    private static final int runsPerConfig=5;//number of independant runs

    //Experiments a-e
    private static final ExperimentConfig[] experiments=new ExperimentConfig[]{
            new ExperimentConfig("a_100c_0m",1.0,0.0),//Experiment a with 100% crossover and 0% mutation
            new ExperimentConfig("b_100c_10m",1.0,0.1),//Experiment a with 100% crossover and 10% mutation
            new ExperimentConfig("c_90c_0m",0.9,0.0),//Experiment a with 90% crossover and 0% mutation
            new ExperimentConfig("d_90c_10m",0.9,0.1),//Experiment a with 90% crossover and 10% mutation
            new ExperimentConfig("e_90c_1m",0.9,0.01)//Experiment a with 90% crossover and 1% mutation
    };


    //Main class
    public static void main(String[] args) throws Exception{

        System.out.println("\n");

        //For each data file
        for(String dataFile:dataFiles){
            if(!Files.exists(Paths.get(dataFile))){
                System.err.println("Data file not found: "+dataFile);
                continue;
            }

            //Read data file (first token is max key length)
            String fileContents=new String(Files.readAllBytes(Paths.get(dataFile)));
            String[] tokens=fileContents.split("\\s+",2);
            if(tokens.length<2){
                System.err.println("Unable to parse data file: "+dataFile);
                continue;
            }
            int maxKeyLen=Integer.parseInt(tokens[0].trim());

            //Trim the encrypted text to letters only (lowercase)
            String encryptedTextRaw=tokens[1];
            String encryptedText=encryptedTextRaw.replaceAll("[^A-Za-z]","").toLowerCase();

            System.out.println("Data file: "+dataFile+" (max key length = "+maxKeyLen+")");

            //Uniform and one-point crossover types
            for(CrossoverType crossType:new CrossoverType[] {CrossoverType.uniform,CrossoverType.onePoint}){
                System.out.println("\nCrossover: "+ crossType.name());

                //For each experiment configuration
                for(ExperimentConfig cfg : experiments){
                    System.out.println("\nRunning experiment "+cfg.name+" (Crossover Rate="+(cfg.crossoverRate)*100+"% Mutation Rate="+(cfg.mutationRate)*100+"%) with "+runsPerConfig+" runs:");

                    //Collect per-run results
                    List<Double> finalBestFitnesses=new ArrayList<>();
                    List<String> finalBestKeys=new ArrayList<>();
                    List<String> finalDecrypted=new ArrayList<>();
                    List<double[]> bestFitnessList=new ArrayList<>(); //One array per run
                    List<double[]> avgPopFitnessList=new ArrayList<>();

                    for(int run=0;run<runsPerConfig;run++){
                        long seed=System.currentTimeMillis()+run*7919L+cfg.name.hashCode();
                        Random rng=new Random(seed);

                        GeneticAlgorithm GA=new GeneticAlgorithm(maxKeyLen,encryptedText,rng);
                        GA.popSize=defaultPopsize;
                        GA.maxGenerations=defaultMaxGen;
                        GA.crossoverRate=cfg.crossoverRate;
                        GA.mutationRate=cfg.mutationRate;
                        GA.tournamentK=defaultTourn;
                        GA.elitismCount=defaultElite;
                        GA.useUniformCrossover=(crossType==CrossoverType.uniform);

                        //Run and capture per-gen stats
                        GARunResult result=GA.runWithStats();

                        finalBestFitnesses.add(result.best.fitness);
                        finalBestKeys.add(result.best.getKeyString());
                        finalDecrypted.add(Evaluation.decrypt(result.best.getKeyString(),encryptedText));
                        bestFitnessList.add(result.bestFitnessPerGen);
                        avgPopFitnessList.add(result.avgPopFitnessPerGen);

                        System.out.printf(" Run %d seed=%d BestFitness=%.6f key=%s%n",run+1,seed,result.best.fitness,result.best.getKeyString());
                    }

                    //Compute aggregated stats across runs
                    double min=Collections.min(finalBestFitnesses);
                    double max=Collections.max(finalBestFitnesses);
                    double mean=mean(finalBestFitnesses);
                    double std=stddev(finalBestFitnesses,mean);
                    double median=median(finalBestFitnesses);

                    System.out.println("\nSummary for "+dataFile+" | "+crossType.name()+" | "+cfg.name);
                    System.out.printf(" final fitness across %d runs: min=%.6f  max=%.6f  mean=%.6f  median=%.6f  std=%.6f%n",runsPerConfig,min,max,mean,median,std);

                    //Print best run's decrypted text (run with minimal final fitness)
                    int bestRunIdx=argmin(finalBestFitnesses);
                    System.out.println("\nBest run index (1-based): "+(bestRunIdx+1));
                    System.out.println(" Best key: "+finalBestKeys.get(bestRunIdx));
                    System.out.println(" Best fitness: "+finalBestFitnesses.get(bestRunIdx));
                    System.out.println(" Decrypted:");
                    String dec=finalDecrypted.get(bestRunIdx);
                    for(int i=0;i<dec.length();i+=80){
                        int end=Math.min(dec.length(),i+80);
                        System.out.println(dec.substring(i,end));
                    }

                    //Produce CSV of average per-generation metrics
                    String safeDataName=dataFile.replaceAll("\\W+","");
                    String csvName=String.format("results_%s_%s_%s.csv",safeDataName,crossType.name(),cfg.name);
                    writeAveragedGenerationCSV(csvName,bestFitnessList,avgPopFitnessList);
                    System.out.println("\nWrote averaged per-generation CSV: "+csvName);
                    System.out.println("----------------------------------------------------------------------");
                }
            }
            System.out.println("\nFinished experiments for "+dataFile+"\n\n");
        }
        System.out.println("All experiments complete.");
    }

    enum CrossoverType{uniform,onePoint}

    public static class ExperimentConfig{
        public String name;
        public double crossoverRate;
        public double mutationRate;
        public ExperimentConfig(String name,double c,double m){
            this.name=name;
            this.crossoverRate=c;
            this.mutationRate=m;
        }
    }


    //Computes the mean of a list
    private static double mean(List<Double> values){
        double s=0;
        for(double v:values) s+=v;
        return s/values.size();
    }

    //Computes the standard deviation of a list of values
    private static double stddev(List<Double> values,double mean){
        double s=0;
        for (double v:values) s+=(v-mean)*(v-mean);
        return Math.sqrt(s/values.size());
    }

    //Computes the median of a list
    private static double median(List<Double> values){
        List<Double> copy=new ArrayList<>(values);
        Collections.sort(copy);
        int n=copy.size();
        if(n%2==1) return copy.get(n/2);
        return (copy.get(n/2-1) + copy.get(n/2))/2.0;
    }

    //Returns the index of the smallest value in the list
    private static int argmin(List<Double> values){
        int idx=0;
        double best=values.get(0);
        for(int i=1;i<values.size();i++){
            if(values.get(i)<best){
                best=values.get(i);
                idx=i;
            }
        }
        return idx;
    }


    //Writes averaged statistics for a generation to a CSV file
    private static void writeAveragedGenerationCSV(String filename,List<double[]> bestPerGenList,List<double[]> avgPopPerGenList){

        //If there are no runs: don't write anything
        if(bestPerGenList.isEmpty()) return;

        int gens=bestPerGenList.get(0).length;
        try (PrintWriter pw=new PrintWriter(new FileWriter(filename))){
            pw.println("generation,avg_best_fitness,avg_population_fitness");

            //Compute avg best and population fitness for each generation
            for(int g=0;g<gens;g++){
                double sumBest=0;
                double sumAvgPop=0;

                //Get fitness values from all runs for this gen
                for(int r=0;r<bestPerGenList.size();r++){
                    sumBest+=bestPerGenList.get(r)[g];
                    sumAvgPop+=avgPopPerGenList.get(r)[g];
                }

                //Compute averages and write to CSV
                double avgBest=sumBest/bestPerGenList.size();
                double avgPop=sumAvgPop/avgPopPerGenList.size();
                pw.printf("%d,%.8f,%.8f%n",g,avgBest,avgPop);
            }
        } catch(IOException e){
            System.err.println("Failed to write CSV "+filename+": "+e.getMessage());
        }
    }



    //Chromosome: array of chars from 'a'-'z' and '-'. Lower fitness is better
    public static class Chromosome implements Comparable<Chromosome>{
        public char[] genes;
        public double fitness=Double.POSITIVE_INFINITY;

        //Character set: lowercase a-z and '-'
        private static final char[] alphabet;
        static{
            alphabet=new char[27];
            for(int i=0;i<26; i++) alphabet[i]=(char)('a'+i);
            alphabet[26]='-';
        }

        //Constructors
        public Chromosome(int length){
            genes=new char[length];
        }
        public Chromosome(int length,Random rng) {
            this(length);
            randomize(rng);
        }

        //Randomly fill genes w letters from the alphabet
        public void randomize(Random rng) {
            for(int i=0;i<genes.length;i++){
                genes[i]=alphabet[rng.nextInt(alphabet.length)];
            }
        }

        //Create a deep copy of this chromosome
        public Chromosome copy(){
            Chromosome c=new Chromosome(genes.length);
            System.arraycopy(this.genes,0,c.genes,0,genes.length);
            c.fitness=this.fitness;
            return c;
        }

        //Convert into string key
        public String getKeyString(){
            return new String(genes);
        }

        //Mutate a single gene
        public void mutateGene(int idx, Random rng){
            genes[idx]=alphabet[rng.nextInt(alphabet.length)];
        }

        //Sorting fitness (the lower the better)
        public int compareTo(Chromosome o){
            return Double.compare(this.fitness, o.fitness);
        }
    }


    //Holds a collection of chromosomes
    public static class Population{
        public List<Chromosome> individuals;
        public Population(int popSize){
            individuals=new ArrayList<>(popSize);
        }

        //Compute and assign fitness for every chromosome in the population
        public void evaluateAll(String encryptedText){
            for(Chromosome c:individuals){
                c.fitness=Evaluation.fitness(c.getKeyString(),encryptedText);
            }
        }

        //Get the chromosome with the lowest fitness
        public Chromosome getBest(){
            return Collections.min(individuals);
        }

        //Compute average fitness
        public double averageFitness(){
            double s=0;
            for(Chromosome c:individuals) s+=c.fitness;
            return s/individuals.size();
        }

        //Sort the pop from best to worst
        public void sortByFitness(){
            Collections.sort(individuals);
        }

        //Tournament selection
        public Chromosome tournamentSelect(int k,Random rng){
            Chromosome best=null;
            int n=individuals.size();
            for(int i=0;i<k;i++){
                Chromosome cand=individuals.get(rng.nextInt(n));
                if(best==null||cand.fitness<best.fitness){
                    best=cand;
                }
            }
            return best;
        }
    }


    //Stores results for a single GA run with its stats
    public static class GARunResult{
        public Chromosome best;
        public double[] bestFitnessPerGen;
        public double[] avgPopFitnessPerGen;
    }


    //Performs genetic algorithm steps
    public static class GeneticAlgorithm{
        public int popSize=defaultPopsize;
        public int maxGenerations=defaultMaxGen;
        public double crossoverRate=1.0;
        public double mutationRate=0.01;
        public int tournamentK=defaultTourn;
        public int elitismCount=defaultElite;
        public boolean useUniformCrossover=true;
        public int chromosomeLength;
        public String encryptedText;
        public Random rng;
        public GeneticAlgorithm(int chromosomeLength,String encryptedText,Random rng){
            this.chromosomeLength=chromosomeLength;
            this.encryptedText=encryptedText;
            this.rng=rng;
        }

        //Initialize the starting populations w random chromosomes
        public Population initPopulation(){
            Population pop=new Population(popSize);
            for(int i=0;i<popSize;i++){
                Chromosome c=new Chromosome(chromosomeLength,rng);
                pop.individuals.add(c);
            }
            pop.evaluateAll(encryptedText);
            return pop;
        }

        //Perform crossover between 2 parents to make 2 children
        public Chromosome[] crossover(Chromosome p1,Chromosome p2){
            Chromosome c1=p1.copy();
            Chromosome c2=p2.copy();
            if(useUniformCrossover){

                //Uniform: Each gene has 50% chance of swapping between parents
                for(int i=0;i<chromosomeLength;i++){
                    if(rng.nextDouble()<0.5){
                        c1.genes[i]=p1.genes[i];
                        c2.genes[i]=p2.genes[i];
                    }else{
                        c1.genes[i]=p2.genes[i];
                        c2.genes[i]=p1.genes[i];
                    }
                }
            } else {
                //One-point: choose random cut point and swap the tails
                int point=rng.nextInt(chromosomeLength);
                for(int i=point;i<chromosomeLength;i++){
                    c1.genes[i]=p2.genes[i];
                    c2.genes[i]=p1.genes[i];
                }
            }
            return new Chromosome[]{c1,c2};
        }

        //Mutate a chromosome
        public void mutate(Chromosome c){
            for(int i=0;i<chromosomeLength;i++) {
                if(rng.nextDouble()<mutationRate) c.mutateGene(i,rng);
            }
        }

        //Run GA and collect per-generation stats
        public GARunResult runWithStats(){
            Population pop=initPopulation();
            Chromosome bestOverall=pop.getBest().copy();

            int gens=maxGenerations+1; //Include generation 0
            double[] bestPerGen=new double[gens];
            double[] avgPopPerGen= new double[gens];

            //Initial stats (generation 0)
            bestPerGen[0]=pop.getBest().fitness;
            avgPopPerGen[0]=pop.averageFitness();

            //Main GA loop
            for(int gen =1;gen<=maxGenerations; gen++){
                Population newPop=new Population(popSize);

                //Sort population by fitness and carry over elite individuals
                pop.sortByFitness();
                for(int e=0;e<Math.min(elitismCount,popSize);e++){
                    newPop.individuals.add(pop.individuals.get(e).copy());
                }

                //Produce new offspring
                while (newPop.individuals.size()<popSize){
                    Chromosome p1=pop.tournamentSelect(tournamentK,rng);
                    Chromosome p2=pop.tournamentSelect(tournamentK,rng);
                    Chromosome child1,child2;
                    if(rng.nextDouble()<crossoverRate){

                        //Perform crossover
                        Chromosome[] kids=crossover(p1,p2);
                        child1=kids[0];
                        child2=kids[1];

                    //If no crossover, copy the parents directly
                    } else{
                        child1=p1.copy();
                        child2=p2.copy();
                    }

                    //Mutate the children and add to new population
                    mutate(child1);
                    if(newPop.individuals.size()<popSize){
                        newPop.individuals.add(child1);
                    }
                    if(newPop.individuals.size()<popSize) {
                        mutate(child2);
                        newPop.individuals.add(child2);
                    }
                }

                //Evaluate pop and update their stats
                newPop.evaluateAll(encryptedText);
                pop=newPop;
                Chromosome genBest=pop.getBest();

                //Best solution seen so far
                if(genBest.fitness<bestOverall.fitness){
                    bestOverall=genBest.copy();
                }

                //Record the per-generation stats
                bestPerGen[gen]=genBest.fitness;
                avgPopPerGen[gen]=pop.averageFitness();
            }

            //Package the final results
            GARunResult r=new GARunResult();
            r.best=bestOverall;
            r.bestFitnessPerGen=bestPerGen;
            r.avgPopFitnessPerGen=avgPopPerGen;
            return r;
        }
    }
}
