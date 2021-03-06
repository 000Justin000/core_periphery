#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <iostream>
#include <math.h> 
#include <fstream>
#include <limits>
#include <ctime>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
//#include <gsl/gsl_rng.h>  //random number generator for initialization.

using namespace std;

const string ntkName = "livejournal1155627";
const int N = 1155627; //number of nodes
const int M = 7188907; //number edges
const int K = 2; //number of groups
const int max_ite = 50; //number of maximum iteration

//Variable declaration.
long int *A[N]; //adjacency list of the network.
long int LastEmpty[N]; //
long int Degree[N];
long int EdgeList[M][2]; //edgelist

long double Gamma[K];
long double Omega[K][K];
long double One_point[N][K]; //one point marginal
map<pair<int,int>, array<array<long double,2>, 2> > Two_point; //two point marginal
map<pair<int,int>, array<long double,2> > Cavity; //message matrix, stored the edges as keys
set<pair<int,int> > EdgeSet;

double External_field[K]; //external field term as sum of logs

int Glist[N]; // core-periphery assignment array.

//Function declaration.
void read_edgelist();
void update_Gamma();
void update_Omega();
void initilize_params();
void compute_external();
void predict();
void compute_one_point();
void compute_two_point();
void compute_cavity();

//main driver function.
int main()
{     
    long int i,j;
    int ite = 0;
    ofstream output;
    string outfname = ntkName + "_glist.txt"; //output file name

    //read file and initialization
	read_edgelist();	
	initilize_params();  

    //main loop
    for (ite = 0; ite < max_ite; ite++)
    {
        //update message and two point marginal
        compute_external();                
        compute_one_point();      
        compute_cavity();

        //update parameters
        compute_two_point();
        update_Omega();
        update_Gamma();    

        // print_info
        if (ite == 0)
        {
            printf("------------------------------------------\n");
            printf(" ite            gamma_0            gamma_1\n");
            printf("------------------------------------------\n");
        }

        printf("%4d  | %15.7Lf    %15.7Lf\n", ite+1, Gamma[0], Gamma[1]);

        if (isnan(Gamma[0]) || isnan(Gamma[1]))
            break;
    }
    printf("------------------------------------------\n\n");

    printf("------------------------------------------\n");
    printf("mixing matrix         1                  2\n");
    printf("------------------------------------------\n");
    printf("   1  | %15.7Lf    %15.7Lf\n", Omega[0][0], Omega[0][1]);
    printf("   2  | %15.7Lf    %15.7Lf\n", Omega[1][0], Omega[1][1]);
    printf("------------------------------------------\n\n");

	predict();

    //compute the objective function
    long double objective_func = 0.0;
    for (i = 0; i < N; i++)
    {
        for (j = i+1; j < N; j++)
        {
            objective_func += (EdgeSet.find(make_pair(i,j)) != EdgeSet.end()) ? 
                              (log(Omega[Glist[i]][Glist[j]])) :
                              (log(1 - Omega[Glist[i]][Glist[j]]));
        }
    }
    printf("------------------------------------------\n");
    printf("objective function         %15.7Lf\n", objective_func);
    printf("------------------------------------------\n");

    //save output assignment to file
    output.open(outfname);
    for(i = 0; i < N; i++)
    {
        output << Glist[i] << endl;
    }    
    output.close();    
}

//based on the implementation by Brian Karrer.
void read_edgelist()
{
     long int i,j;
     ifstream InputFile;     
    
     string fileName = ntkName + ".txt";     //change file name here
     string lineread;
     char *buffer;
     long int entry1 = 0;     
     long int entry2 = 0;
     int counter = 0;     
     long int ignore = 0;     	

	InputFile.open(fileName.c_str());
	if (!InputFile)
	{
		cout << "Error in opening file";
		cin.get();
		return;
	}     	

     while(getline(InputFile, lineread)) // Read line by line
     {
     buffer = new char [lineread.size()+1];
     strcpy(buffer, lineread.c_str());
     sscanf(buffer, "%zu %zu %zu", &entry1, &entry2, &ignore);
         
     EdgeList[counter][0] = entry1;
     EdgeList[counter][1] = entry2;

     EdgeSet.insert(make_pair(min(entry1,entry2), max(entry1,entry2)));
    
     counter = counter+1;
 
     delete[] buffer;
     }
     InputFile.close();	         

	 // We start the degree values and LastEmpty all at zero
     for(i=0; i < N; i++)
     {
        Degree[i] = 0;
        LastEmpty[i] = 0;
     }
     
     // First we count the degrees by scanning through the list once
     for(i=0; i < M; i++)
     {
         Degree[EdgeList[i][0]]++;
         Degree[EdgeList[i][1]]++;
     }
	     
     // Now we make space in the adjacency lists as well as the 
     for(i=0; i < N; i++)
     {
     A[i] = new long int [Degree[i]];
     }

     // Now we read the edges into the adjacency lists utilizing last empty to put them into
     // the proper spots; 
     for(i=0; i < M; i++)
     {        
       A[EdgeList[i][0]][LastEmpty[EdgeList[i][0]]] = EdgeList[i][1];       
       LastEmpty[EdgeList[i][0]]++;              	
       A[EdgeList[i][1]][LastEmpty[EdgeList[i][1]]] = EdgeList[i][0];
       LastEmpty[EdgeList[i][1]]++;
     }                
     return;     
}

//update the parameters based on current belief
void update_Gamma()
{
    int r;
    long int i;
    long double NORM=0;    

    fill(Gamma, Gamma + K, 0);
    for(r = 0; r < K; r++)
    {
        for(i = 0; i < N; i++)
        {
            Gamma[r] += One_point[i][r]; 
        }
        NORM += Gamma[r];        
    }
    
    //normalize
    for(r=0; r < K; r++)
    {
        Gamma[r] = Gamma[r]/NORM;
    }    

    if (Gamma[0] > 0.99)
    {
        Gamma[0] = 0.99;
        Gamma[1] = 0.01;
    }

    if (Gamma[1] > 0.99)
    {
        Gamma[0] = 0.01;
        Gamma[1] = 0.99;
    }

	return;
}

void update_Omega()
{
    long int i,j;
    int r,s,it;
    long double SIGMA;

    for(r = 0; r < K; r++)
    {
        for(s = 0; s < K; s++)
        {
            SIGMA = 0;
            for(i = 0; i < N; i++)
            {
                for(it = 0; it < Degree[i]; it++)
                {
                    j = A[i][it];
                    SIGMA += Two_point[make_pair(i,j)][r][s];
                }
            }
            //update the mixing matrix.
            Omega[r][s] = SIGMA/(double(N)*double(N)*Gamma[r]*Gamma[s]);
        }
    } 

	return;
}

//make prediction based on belief. i,e. assign node to the group with highest belief.
void predict()
{   
    long int i;
    for (i = 0; i < N; i++)
    {
        Glist[i] = distance(One_point[i],max_element(One_point[i],One_point[i]+K));        
    }    
	return;
}

void initilize_params()
{
    long int i,j;
    int r;
	//The initialization can be achieved many ways and different initialization 
	//will result in possibly different results. Try different initialization 
	//point as well as initialization method for better results. Here the 
	//method is the 'apriori' method where we have a rough idea of what the parameters
	//should be.
	srand(time(NULL)); //random seed 
	Gamma[0] = ((long double) rand() / (RAND_MAX));
	Gamma[1] = 1.0 - Gamma[0];
    //Omega[0][0] = 0.1;
    //Omega[0][1] = 0.03;
    //Omega[1][0] = 0.03;
    //Omega[1][1] = 0.01;
    Omega[0][0] = double(M) / (double(N)*double(N)) * 2;
    Omega[0][1] = double(M) / (double(N)*double(N));
    Omega[1][0] = double(M) / (double(N)*double(N));
    Omega[1][1] = double(M) / (double(N)*double(N)) / 2;

	//one could randomly generate a,b repeatedly for each node/edge. one value usually work just as well.
	long double a;		
	long double b;
	a = ((long double) rand() / (RAND_MAX));
	b = ((long double) rand() / (RAND_MAX));	

    //initialize one point marginal to unbalanced belief.
    for(i = 0; i < N ;i++)
    {
       One_point[i][0] = a;
       One_point[i][1] = 1- a;
    }	    
           
    //initialize the message on the edges; also initialize the two point marginal to zeros.
    for(i = 0; i < M; i++)
    {
        for(r = 0; r < K; r++)
        {            
            Cavity[make_pair(EdgeList[i][0],EdgeList[i][1])][r] = r==0 ? b:1-b;
            Cavity[make_pair(EdgeList[i][1],EdgeList[i][0])][r] = r==0 ? b:1-b;
            Two_point[make_pair(EdgeList[i][1],EdgeList[i][0])] = {{{0,0},{0,0}}};
            Two_point[make_pair(EdgeList[i][0],EdgeList[i][1])] = {{{0,0},{0,0}}};            
        }        
    }    
	return;	
}

//compute the external field term 
void compute_external()
{
    long int i;
    int r,s;
    long double SIGMA;

    fill(External_field,External_field+K,0);
    
    for(r = 0; r < K; r++)
    {
        for(i=0; i < N; i++)
        {
            SIGMA = 0;
            for(s = 0; s < K; s++)
            {
                SIGMA += One_point[i][s]*exp(-Omega[r][s]);
            }       
            External_field[r] += log(SIGMA);                  
        }
    }           
    return;      
}

//compute the one point marginal
void compute_one_point()
{
    long double BUFFER[K];
    long int i,j;
    int r,s,it;
    long double SIGMA1,SIGMA2,PROD;
    
    for(i = 0; i < N; i++)
    {
        fill(BUFFER,BUFFER+K,0);
        for(r = 0; r < K; r++)
        {
            PROD = 0;
            for(it = 0; it < Degree[i]; it++)
            {
                j = A[i][it];
                SIGMA1 = 0;
                SIGMA2 = 0;
                for(s = 0; s < K; s++)
                {                       
                    SIGMA1 += Cavity[make_pair(j,i)][s]*Omega[r][s]*exp(-Omega[r][s]);
                    SIGMA2 += One_point[j][s]*exp(-Omega[r][s]);                    
                }                
                PROD += log(SIGMA1)- log(SIGMA2);                    
            }
            BUFFER[r] = log(Gamma[r]) + PROD + External_field[r];
        }
        //normalize and set the value
        long double x = 1/(exp(BUFFER[0]-BUFFER[1])+1);        
        One_point[i][0] = 1 -x;
        One_point[i][1] = x;        
    }
        
    return;
}

//compute the two point marginal
void compute_two_point()
{       
    long double BUFFER[K][K];
    long double NORM,PROD; //normalization term, sum of the individual q_{ij}^{rs}.
    long int i,j;
    int r,s,it;

    for(i = 0; i < N; i++)
    {
        for(it = 0; it < Degree[i]; it++)
        {
            j = A[i][it];
            NORM = 0;
            fill(BUFFER[0],BUFFER[0]+K,0);
            fill(BUFFER[1],BUFFER[1]+K,0);
            //summing over the four possible values of two point marginal
            for(r = 0; r < K; r++)
            {
                for(s = 0; s < K; s++)
                {
                    PROD = Omega[r][s]*exp(Omega[r][s])*Cavity[make_pair(i,j)][r]*Cavity[make_pair(j,i)][s];
                    NORM += PROD;
                    BUFFER[r][s] = PROD;                    
                }
            }
            
            //normalizing the two point marginal so they sum to unity
            for(r = 0; r < K; r++)
            {
                for(s = 0; s < K; s++)
                {
                    Two_point[make_pair(i,j)][r][s] = BUFFER[r][s]/NORM;
                }
            }                           
        }
    }
    return;
}

//compute the message matrix
//only update the edge terms, the non-egde terms will just be the value of the one point marginal
void compute_cavity()
{
    long double BUFFER[K];
    long int i,j,k;
    int r,s,it1,it2; //use two iterators for two edges
    long double SIGMA1,SIGMA2,PROD;

    for(i = 0; i < N; i++)
    {
        for(it1 = 0; it1 < Degree[i]; it1++)
        {
            j = A[i][it1];
            for(r = 0; r < K; r++)
            {
                PROD = 0;
                for(it2 = 0; it2 < Degree[i]; it2++)
                {
                    k = A[i][it2];
                    if(k != j)
                    {   
                        SIGMA1 = 0;
                        SIGMA2 = 0;
                        for(s = 0; s < K; s++)
                        {
                            SIGMA1 += Cavity[make_pair(k,i)][s]*Omega[r][s]*exp(-Omega[r][s]);
                            SIGMA2 += One_point[k][s]*exp(-Omega[r][s]);
                        }
                        PROD += log(SIGMA1) - log(SIGMA2);
                    }                                        
                }
                BUFFER[r] = log(Gamma[r]) + PROD + External_field[r];
            }
            //normalize
            long double x = 1/(exp(BUFFER[0]-BUFFER[1])+1);        
            Cavity[make_pair(i,j)][0] = 1 - x;
            Cavity[make_pair(i,j)][1] = x;            
        }
    }
    return;
}
