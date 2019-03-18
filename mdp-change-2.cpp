#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>
#include "randgen.h"


using namespace std;
float b1 = 0.2,b2 = 0.3;  // 1st node data generation co-eff
float d1 = 0.3,d2 = 0.2;  // 2nd node data generation co-eff
float c1 = 0.5;   // energy

// Buffer sizes
const int EMAX = 20,DMAX = 10;
const int baseD1 = DMAX+1;
const int baseE = EMAX+1;
const int baseD2 = 2*DMAX+1;

// Maximum value of energy arrival
const int Max_Energy = 10;
// Maximum value of data arrival
const int Max_Data = 4;

// Cost variables
double reward = 0.0;
double dCost = 0.0;

//Rates for noise variable for data and energy generation
double lambda_x1 = 1,lambda_x2 = 1;
double lambda_y = 20;

// Number of steps taken
long nStep = 0;

// state variables
int qk1 = 0,qk2 = 0, E = 0,x1 = 0,x2 = 0,eY = 0;
int s1 = 0, s2 = 0, s3 = 0,x1P = 0, x2P = 0, eYP = 0;

// usual energy split
int a1 = 0, a2 = 0;

// Choose the algorithm to run
int algo = 0;

// Files to store the outputs - avgcost and policy
ofstream resultsFile,policyFile;

// Mapping states to actions available
map<int,vector<vector<int> > > hmap;
vector< vector<int> > acList;

// Stores the minimum action index
int minActionIndex = 0;

//discount parameter for q-learning
float discountF = 0.9;

// Stepsizes
const double alpha_stepsize = 0.1;
const double epsilon = 0.3;

// To learn or evaluate ?
bool freezeLearning = false;

// Number of iterations required to evaluate policy
long evalSteps = 20000000;

// Number of iterations required to learn policy
long learnSteps = 700000000;


// Data structures used for combined nodes q-learning
int A = 0; //total amount of energy to be split further - cnql and greedy
float cnQ[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][baseE]= {0.0};
int cnP[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Energy+1] = {0};
bool cnV[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][baseE] = {false};

// Data structures used for q-learning 
float Q[baseD1][baseD1][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][baseE][baseE] = {0.0};

int policy[baseD1][baseD1][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][2] = {0}; // saves high, low or medium action
int vcount[baseD1][baseD1][baseE][Max_Data+1][Max_Data+1][Max_Energy+1] = {0};

bool isvisited[baseD1][baseD1][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][baseE][baseE] = {false};

// Data structures for UCB q learning
int savcount[baseD1][baseD1][baseE][Max_Data+1][Max_Data+1][Max_Energy+1][baseE][baseE] = {0};

//for ucb exploration
double alpha_ucb = 1;


// Random number generator
randgen* generator;

// Function declarations
double convertEnergytoData(int);
void learnCost_updatestate();
void evaluateCost_UpdateState();
void updateState(int,int);
void generateActionSpace();
void simulate();
void updateDCost();
int generateData(int);
int harvestEnergy();
int generatePoissonSample(int);
void saveResult();

void beGreedy();
void QLearning();
void getSplit();

//q-learning specific functions
void getMinAction();
void savePolicy();
void suggestAction();
void prevState_QUpdate();
void resetDataStructures();
void cnql_savepolicy();
void eql_savepolicy();

int main(int argc,char*argv[])
{
	generator = new randgen();
	if (argc < 3)
	{
	    cout << "\nUsage: *.out <Node1 noise rate> <algo-start-index>\n" << endl;
	    exit(0);
	}
	lambda_x1 = atof(argv[1]);
	algo = atoi(argv[2]);
	cout << "lambda_x1 = "<<lambda_x1 << ", lambda_x2 = " <<lambda_x2 << endl;
	cout << "lambda_y = "<<lambda_y <<endl;
	cout << "DMAX = "<< DMAX<< ", EMAX = "<< EMAX << endl;
	cout << "Max data arrival: "<< Max_Data << endl;
	cout << "Max energy arrival: " << Max_Energy << endl;
	generateActionSpace();
	//resetDataStructures();

	    switch(algo)
	     {
		case 1: // greedy
		   	resultsFile.open("greedy-avgcost-2nmdp.txt",ofstream::out|ofstream::app);
			cout << "Greedy method"<< endl;
			beGreedy();
			break;
		case 2: //combined-node ql
			resultsFile.open("cnql-dcost-2nmdp.txt",ofstream::out|ofstream::app);
			policyFile.open("cnql-policy-2nmdp.txt",ofstream::out);
			cout << "Combined Nodes method"<< endl;
			QLearning();
			break;
		
		case 3: //epsilon-greedy ql
		        resultsFile.open("qle-avgcost-2nmdp.txt",ofstream::out|ofstream::app);
		        policyFile.open("qle-policy-2nmdp.txt",ofstream::out);	
			cout << "Epsilon greedy Q-Learning"<< endl;		
			QLearning();
			//resetDataStructures();
			break;
		
		case 4: //ucb ql
			resultsFile.open("qlucb-cost-2nmdp.txt",ofstream::out|ofstream::app);
			policyFile.open("qlucb-policy-2nmdp.txt",ofstream::out);
			cout << "UCB Exploration Q-Learning"<< endl;			
			QLearning();
			break;		
	     }	
	cout << "Average cost: "<< dCost <<"\n"<< endl;
	
	delete generator;
	return 0;
}

void init_data()
{
	nStep = 0;
	
	dCost = 0.0; reward = 0.0;

	qk1 = 0 ; qk2 = 0 ; E = 0 ; x1 = 0 ; x2 = 0 ; eY = 0;
	s1 = 0 ; s2 = 0 ; s3 = 0 ; x1P = 0 ; x2P = 0 ; eYP = 0;
	
	A = 0;
	a1 = 0; a2 = 0;
	minActionIndex = 0;
}

void beGreedy()
{
	init_data();
	srand(time(0));
        for (long i = 1 ; i <= evalSteps ; i++)
	    simulate();
	saveResult();
	resultsFile.close();
}

void QLearning()
{
	init_data();
	srand(time(0));
	freezeLearning = false;

	for (long i = 1 ; i <= learnSteps ; i++)
	  { simulate(); }
	
	savePolicy();
	cout << "Policy stored "<< endl;

	init_data();
	freezeLearning = true;

	srand(time(0));
        for (long i = 1 ; i <= evalSteps ; i++) 
	   { simulate(); }

	saveResult();
	resultsFile.close();	
	policyFile.close();
}

void simulate()
{
	switch(algo)
	{
	   case 1: nStep++;
		   evaluateCost_UpdateState();
	           updateDCost();
	   	   getSplit();
		   break;
	   case 2: if (!freezeLearning)
	   	   {	    
	                learnCost_updatestate();
	                prevState_QUpdate();
	       		suggestAction();	  
	       		cnV[qk1+qk2][E][x1][x2][eY][A] = true;
	   	   }
		   else
	   	   {
	       		nStep++;
	       		evaluateCost_UpdateState();
	       		updateDCost();
	       		A = cnP[qk1+qk2][E][x1][x2][eY];
	       		getSplit();
	   	   }	
		   break;
	   
	   case 3: case 4:
		   if (!freezeLearning)
		   {
			learnCost_updatestate();
	                prevState_QUpdate();
	       		suggestAction();
			vcount[qk1][qk2][E][x1][x2][eY]++;
			isvisited[qk1][qk2][E][x1][x2][eY][a1][a2] = true;

			if (algo == 4) savcount[qk1][qk2][E][x1][x2][eY][a1][a2]++;
		   }
		   else
		   {
	       		nStep++;
	       		evaluateCost_UpdateState();
	       		updateDCost();
	                a1 = policy[qk1][qk2][E][x1][x2][eY][0];
	       		a2 = policy[qk1][qk2][E][x1][x2][eY][1];
		   }
		   break;		
	}
	s1 = qk1;
	s2 = qk2;
	s3 = E;
	x1P = x1;
	x2P = x2; 
	eYP = eY;
}

void prevState_QUpdate()
{
	double tmp = 0.0,tmp1 = 0.0;
	getMinAction();
	
	switch (algo)
	{	
	   case 2:
	   	tmp = cnQ[s1+s2][s3][x1P][x2P][eYP][A];
	   	tmp1 = cnQ[qk1+qk2][E][x1][x2][eY][minActionIndex];
	   	tmp = (float)((1-alpha_stepsize)*tmp + alpha_stepsize*(reward + discountF*tmp1));			
	   	cnQ[s1+s2][s3][x1P][x2P][eYP][A] = tmp;
		break;
	   case 3: case 4:
	 	vector<int> u = acList[minActionIndex];
	   	int minA1 = u[0];
	   	int minA2 = u[1];
	   	tmp = Q[s1][s2][s3][x1P][x2P][eYP][a1][a2];
	   	tmp1 = Q[qk1][qk2][E][x1][x2][eY][minA1][minA2];

	   	tmp = (float)((1-alpha_stepsize)*tmp + alpha_stepsize*(reward + discountF*tmp1));
			
	   	Q[s1][s2][s3][x1P][x2P][eYP][a1][a2] = tmp;

		break;
	}
}

void getMinAction()
{	
	switch (algo)
	{
	   case 2:
	   	 minActionIndex = 0;
	   	 for (int i = 1 ; i <= E ; i++)
	   	 {	  
	      		if (cnQ[qk1+qk2][E][x1][x2][eY][i] < cnQ[qk1+qk2][E][x1][x2][eY][minActionIndex])
	   	  		minActionIndex = i;
	   	 }
		 break;
	   case 3:
		 acList = hmap.at(E);
	   	 minActionIndex = 0;

	   	 for (vector<int>::size_type i=1 ; i< acList.size() ; i++)
	   	 {
	      	     vector<int> y = acList[minActionIndex];
	      	     int mA1 = y[0]; int mA2 = y[1];

	      	     vector<int> x = acList[i];
	      	     int cA1 = x[0]; int cA2 = x[1];

	      	     if (Q[qk1][qk2][E][x1][x2][eY][cA1][cA2] < Q[qk1][qk2][E][x1][x2][eY][mA1][mA2])
	   	  	minActionIndex = i;
	   	 }
		 break;
	   case 4:
		 acList = hmap.at(E);
	   	 minActionIndex = 0;

	   	 for (vector<int>::size_type i=1 ; i< acList.size() ; i++)
	     	 {
	             vector<int> y = acList[minActionIndex];
	             int mA1 = y[0]; int mA2 = y[1];
	  	     vector<int> x = acList[i];
	  	     int cA1 = x[0]; int cA2 = x[1];
	  
	  	     int svc = vcount[qk1][qk2][E][x1][x2][eY];
	  	     int savc1 = savcount[qk1][qk2][E][x1][x2][eY][mA1][mA2];
	  	     int savc2 = savcount[qk1][qk2][E][x1][x2][eY][cA1][cA2];

	  	     double val1 = 0.0,val2 = 0.0;
	  	     if (savc1 > 0)  // max{ -Q(s,a) + sqrt(ln(N(s))/N(s,a)) }	    
	     	         val1 = alpha_ucb*( sqrt( log(svc)/savc1 ) ) - Q[qk1][qk2][E][x1][x2][eY][mA1][mA2] ;
	  	     else val1 = 1;

	  	     if (savc2 > 0)  
	                 val2 = alpha_ucb*( sqrt( log(svc)/savc2 ) ) - Q[qk1][qk2][E][x1][x2][eY][cA1][cA2] ;
	  	     else val2 = 1;

	  	     if (val2 > val1)
	   	    	minActionIndex = i;	   	   
	     	}
		break;
	}
}


int generateData(int node) // Includes correlation of data arrival
{
	int x = 0;
	switch(node)
	{
	   case 1: x = (int)floor(b1*x1P + b2*x2P + generatePoissonSample(1));
		break;
	   case 2: x = (int)floor(d1*x1P + d2*x2P + generatePoissonSample(2));
		break;
	}
	return x;
}

int harvestEnergy()
{
	int y = (int)floor(c1*eYP + generatePoissonSample(3));
	return y;
}

int generatePoissonSample(int option)
{
	float lambda = 0.0;
	switch(option)
	{
	  case 1: lambda = lambda_x1;
	  	  break;
	  case 2: lambda = lambda_x2;
		  break;
	  case 3: lambda = lambda_y;
		  break;
	}
	double L = exp(-lambda);
	double p = 1.0;
	int k = 0;
	do{
	   k++;
	   p *= generator->drand();
	} while(p>L);
	return k-1;
}

void updateDCost()
{  dCost = dCost + (reward - dCost)/(nStep+1); }

void evaluateCost_UpdateState()
{
	double gk1 = convertEnergytoData(1);
	double gk2 = convertEnergytoData(2);

	int Gk1 = (int)floor(gk1);
	int Gk2 = (int)floor(gk2);

	double costNode1 = fmax((qk1 - Gk1),0);
	double costNode2 = fmax((qk2 - Gk2),0);

	reward = (costNode1 + costNode2)/DMAX;
	updateState(Gk1,Gk2);
}

void learnCost_updatestate()
{
	double gk1 = convertEnergytoData(1);
	double gk2 = convertEnergytoData(2);

	int Gk1 = (int)floor(gk1);
	int Gk2 = (int)floor(gk2);

	// Precision Error in splitting energy -- for (1,2,x) action (2,y) is better than (3,z)
	double err1 = gk1-Gk1;
	double err2 = gk2-Gk2;	    
	// Find error in the total sum of energy given (wrt Q-Learning on combined nodes)
	// If energy given to a node is more than the required energy 
	double err3 = 0.0,err4 = 0.0;	    
	if (Gk1 > qk1)
  	    err3 = Gk1-qk1;
	if (Gk2 > qk2)
	    err4 = Gk2-qk2;	    

	double costNode1 = fmax((qk1 - Gk1),0) + err1 + err3;
	double costNode2 = fmax((qk2 - Gk2),0) + err2 + err4;

	reward = (costNode1 + costNode2)/DMAX;
	updateState(Gk1,Gk2);
}

void updateState(int Gk1,int Gk2)
{
	int x_k1 = generateData(1);
	int x_k2 = generateData(2);
	int yk = harvestEnergy();

	if (x_k1 > Max_Data) x_k1 = Max_Data;
	if (x_k2 > Max_Data) x_k2 = Max_Data;
	if (yk > Max_Energy) yk = Max_Energy;

	x1 = x_k1;
	x2 = x_k2;
	eY = yk;

	switch(algo)
	{
	   case 1: case 3: case 4: E -= (a1+a2); //greedy, ql,		
		break;
	   case 2: E -= A;	//cnql
		break;
	}
	if (E < 0) E = 0;
	E += eY;
	if (E > EMAX) E = EMAX;

	// Update data queue length
	qk1 = fmax((qk1 - Gk1),0) + x1;
	if (qk1 > DMAX)	qk1 = DMAX;

	qk2 = fmax((qk2 - Gk2),0) + x2;
	if (qk2 > DMAX) qk2 = DMAX;	
}


double convertEnergytoData(int node)
{
	if (node == 1) 
	    return ( 2*log(1+a1) );
	else 
            return ( 2*log(1+a2) );
}

void getSplit()// called by greedy and cnql methods
{
	if (algo == 1) A = E;	//if greedy, then use entire energy
	if (A > 0)
	{
	   vector<double> cdf;
	   int req1 = (int)ceil(exp(qk1/2.0) - 1);
	   int req2 = (int)ceil(exp(qk2/2.0) - 1);
	   int sum1 = 0, sum2 = 0;
	
	   vector<int> nonFull;
	  
	   if (qk1 > 0 && !(sum1 == req1))								
		nonFull.push_back(1);
			
	   if (qk2 > 0 && !(sum2 == req2))								
		nonFull.push_back(2);

	   for (int k = 1; k <= A && !nonFull.empty(); k++)
	    {
		double denom = 0.0,numerator = 0.0;
		cdf.clear();
		for (vector<int>::size_type it = 0 ; it < nonFull.size(); it++)
		 {
		    switch(nonFull[it])
		     {
			case 1: denom += qk1;
				numerator += qk1;
				cdf.push_back(numerator);
				break;
			case 2: denom += qk2;
				numerator += qk2;
				cdf.push_back(numerator);
				break;
		     }
		 }
		int picked = 0;
		double pick = generator->drand();
		for (vector<double>::size_type i = 0; i < cdf.size(); )
		  {
		     cdf[i] = cdf[i]/denom;
		     if (pick > cdf[i])
			 i++;
		     else
		      {
			picked = i;
			break;
		      }
		  }		
		switch(nonFull[picked])
  		  {
		    case 1:sum1++;
			   if (sum1 == req1) 
			      nonFull.erase(nonFull.begin()+picked);
			   break;
		    case 2:sum2++;
			   if (sum2 == req2)
			      nonFull.erase(nonFull.begin()+picked);
			   break;
		  }
	    }
	   a1 = sum1; a2 = sum2;  	   
	}
	else
	{ a1 = 0; a2 = 0; }
	
}

void generateActionSpace()
{
	for (int en = 0 ; en <= EMAX ; en++){			
	    vector<vector<int> > alist;
	    for (int t = en ; t >= 0 ; t--){
		for (int i = t ; i >= 0 ; i--){
		    vector<int> arr(2,0);
		    arr[0] = i;
		    arr[1] = t-i;		
		    alist.push_back(arr);
		}
    	    }
	hmap.insert(make_pair(en,alist));
        }
}


void suggestAction()
{
	double pick = generator->drand();
	vector<int> y;
	switch(algo)
	{
	   case 2:if (pick > epsilon) //EXPLOIT
	    	    A = minActionIndex;		
		  else	
	    	    A = rand() % (E+1);
		  getSplit();
		  break;	
	   case 3:
		  if (pick > epsilon) //EXPLOIT
	    	     y = acList[minActionIndex];		
		  else
		    {	
			int nongreedy = rand() % (acList.size());
	    		y = acList[nongreedy]; 
		    }
		  a1 = y[0]; a2 = y[1];	
		break;
	   case 4:vector<int> u = acList[minActionIndex]; 
	   	  a1 = u[0]; a2 = u[1];//minActionIndex determined in getMinAction()	  	  
		break;

	}
}

void savePolicy()
{
	switch (algo)
	{
	    case 2: cnql_savepolicy();
		break;
	    case 3: case 4: eql_savepolicy();
		break;
	}
	policyFile.flush();
}

void cnql_savepolicy()
{
	for (int state = 0 ; state < baseD2 ; state++)
	{
	   for (int energy = 0 ; energy <= EMAX ; energy++)
	   {
 	     for (int xk1 = 0 ; xk1 <= Max_Data ; xk1++)
	     { 
 		for (int xk2 = 0 ; xk2 <= Max_Data ; xk2++)
		{
		   for (int yk = 0; yk <= Max_Energy ; yk++)
		   {		  
		        int j = 0;
	     	        for ( ; j <= energy && !cnV[state][energy][xk1][xk2][yk][j]; j++) ;
	     		int ma = 0;
			if (j <= energy) 
	     		{
			   ma = j;
			   for (int k = ma+1; k <= energy ; k++)
			   {
if ( (cnQ[state][energy][xk1][xk2][yk][k] < cnQ[state][energy][xk1][xk2][yk][ma]) && cnV[state][energy][xk1][xk2][yk][k] )
		 	   		ma = k;
			   }
	     		}
	     		cnP[state][energy][xk1][xk2][yk] = ma;
	     		policyFile << "\n State: " << state << ", "<< energy <<", "<< xk1 <<", " << xk2 <<
			", "<< yk << "\t Action: "<< ma ;
		   }
	        }
	     }
	   }
	}
}

void eql_savepolicy()
{
	for (int en = 0 ; en < baseE ; en++)
	{
	    acList = hmap.at(en);		
	    for (int q1 = 0; q1 < baseD1 ; q1++)
	     {
		for (int q2 = 0; q2 < baseD1 ; q2++)
		{
		  for (int xk1 = 0 ; xk1 <= Max_Data ; xk1++)
		  {
		      for (int xk2 = 0; xk2 <= Max_Data ; xk2++)
		      {
			  for (int ek = 0; ek <= Max_Energy ; ek++)
			  {
		  	      vector<int> indices;
		  	      for (vector<int>::size_type y = 0; y < acList.size(); y++)
		    	      {
		        	vector<int> x = acList[y];
				int cA1 = x[0]; int cA2 = x[1];
				if (Q[q1][q2][en][xk1][xk2][ek][cA1][cA2] > 0.0 || isvisited[q1][q2][en][xk1][xk2][ek][cA1][cA2])
			    		indices.push_back(y);			
		    	      }
		   	      minActionIndex = 0;
		   	      for (vector<int>::size_type u = 1; u < indices.size() ; u++)
		   	      {
		      		int indexM = indices[minActionIndex];
		      		int indexC = indices[u];						
						
		      		vector<int> u1 = acList[indexM];
		      		int mA1 = u1[0]; int mA2 = u1[1];
						
		      		vector<int> u2 = acList[indexC];
		      		int cA1 = u2[0]; int cA2 = u2[1];
						
		      		if (Q[q1][q2][en][xk1][xk2][ek][cA1][cA2] < Q[q1][q2][en][xk1][xk2][ek][mA1][mA2])
			  		minActionIndex = u;	
		   	      }
		   	      if (indices.size() > 0)
		   	      {	
		      		int indexM = indices[minActionIndex];	
		      		vector<int> z = acList[indexM];	 
		      		policy[q1][q2][en][xk1][xk2][ek][0] = z[0];
		      		policy[q1][q2][en][xk1][xk2][ek][1] = z[1];

		      		policyFile << "\n State: " << q1 << "," << q2 <<"," << en <<", "<<xk1 <<", "<<xk2
				<< ", "<<ek << "   Action: "<< z[0] << "," << z[1];
		   	      }
		   	      else 
		   	      {
				policy[q1][q2][en][xk1][xk2][ek][0] = 0;
				policy[q1][q2][en][xk1][xk2][ek][1] = 0;

				policyFile << "\n State: " << q1 << "," << q2 <<"," << en <<", "<<xk1 <<", "<<xk2 
				<<","<< en <<"   Action: 0,0";
		   	      }
			}
		     }
		  }
		}
	     }
	}
}


void saveResult()
{
	resultsFile <<"\n-------------------------------------------------------\n";
	resultsFile << "DMAX = " <<DMAX << ", EMAX = "<<EMAX << "\n";
	resultsFile << "lambda_x1 = " <<lambda_x1 << ", lambda_x2 = " <<lambda_x2 ;
	resultsFile << "\nlambda_y = " <<lambda_y ;

	if (algo > 1) resultsFile << "\n Learning steps: " <<learnSteps;
	resultsFile << "\n Evaluation steps: " <<evalSteps;
	resultsFile << "\n Max_data: "<< Max_Data;
	resultsFile << "\n Max_energy: " << Max_Energy;

	if (algo == 2 || algo ==3) resultsFile << "\n Epsilon = "<<epsilon;
	resultsFile <<"\nAverage cost: "<< dCost;	
}
