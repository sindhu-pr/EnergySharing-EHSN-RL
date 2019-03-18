#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>
#include "randgen.h"

using namespace std;

// Buffer sizes
const int EMAX = 25,DMAX = 10;
const int baseE = EMAX+1;
const int baseD2 = 4*DMAX + 1;

// number of partitions of data and energy buffers
const int nlE = 3;
const int nlD = 2;

//manual partitions - Energy and data threshold levels
int DL1low = 0, DL1up = 5;
int DL2low = 6, DL2up = 10;
//int DL3low = 11, DL3up = 14;

int EL1low = 0, EL1up = 7;
int EL2low = 8, EL2up = 20;
int EL3low = 21, EL3up = 25;

float b1 = 0.1,b2 = 0.1,b3 = 0.1, b4 = 0.2;  // 1st node data generation co-eff
float d1 = 0.1,d2 = 0.1,d3 = 0.2, d4 = 0.1;  // 2nd node data generation co-eff
float f1 = 0.1,f2 = 0.2,f3 = 0.1, f4 = 0.1;  // 
float h1 = 0.2,h2 = 0.1,h3 = 0.1, h4 = 0.1;  // 
float c1 = 0.5;   // energy

// Maximum value of energy arrival
const int Max_Energy = 20;
// Maximum value of data arrival
const int Max_Data = 4;

// Cost variables
double reward = 0.0;
double dCost = 0.0;

//Rates for noise variable for data and energy generation
double lambda_x1 = 1,lambda_x2 = 1, lambda_x3 = 1, lambda_x4 = 1;
double lambda_y = 5;

// Number of steps taken
long nStep = 0;

int qk1 = 0, qk2 = 0, qk3 = 0, qk4 = 0, E = 0;
int qk1P = 0, qk2P = 0, qk3P = 0, qk4P = 0, EP = 0;

int x1 = 0,x2 = 0,x3 = 0, x4 = 0, eY = 0;
int x1P = 0, x2P = 0, x3P = 0, x4P = 0, eYP = 0;

// usual energy split
int a1 = 0, a2 = 0, a3 = 0, a4 = 0;

//says whether an action is low, medium or high (aggregate action)
int agA1 = 0, agA2 = 0, agA3 = 0, agA4 = 0;
int agAP1 = 0, agAP2 = 0, agAP3 = 0, agAP4 = 0;

// Stores the aggregate state
int as1 = 0,as2 = 0,as3 = 0, as4 = 0, as5 = 0;
int asP1 = 0, asP2 = 0, asP3 = 0, asP4 = 0, asP5 = 0;

// Choose the algorithm to run
int algo = 0;

// Files to store the outputs - avgcost and policy
ofstream resultsFile,policyFile;

// partition-state map
map<int,int> fsMap,dfsMap;

// Mapping aggregate states to aggregate actions available
map<int,vector<vector<int> > > fHmap;
vector< vector<int> > acList;

// Stores the minimum action index
int minActionIndex = 0;

//discount parameter for q-learning
float discountF = 0.9;

// Stepsizes
double alpha_stepsize = 0.1;
double epsilon = 0.3;

// To learn or evaluate ?
bool freezeLearning = false;

// Number of iterations required to evaluate policy
long evalSteps = 20000000;

// Number of iterations required to learn policy
long learnSteps = 700000000;

// Data structures used for combined nodes q-learning
int A = 0; //total amount of energy to be split further - cnql and greedy
float cnQ[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][baseE]= {0.0};
int cnP[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1] = {0};
bool cnV[baseD2][baseE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][baseE] = {false};

// Data structures used for q-learning 
float Q[nlD][nlD][nlD][nlD][nlE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][nlE][nlE][nlE][nlE] = {0.0};

int policy[nlD][nlD][nlD][nlD][nlE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][4] = {0};
int vcount[nlD][nlD][nlD][nlD][nlE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1] = {0};

bool isvisited[nlD][nlD][nlD][nlD][nlE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][nlE][nlE][nlE][nlE] = {false};

//for ucb exploration
double alpha_ucb = 1;
int savcount[nlD][nlD][nlD][nlD][nlE][Max_Data+1][Max_Data+1][Max_Data+1][Max_Data+1][Max_Energy+1][nlE][nlE][nlE][nlE] = {0};

// Random number generator
randgen* generator;

// Function declarations

double convertEnergytoData(int);
void learnCost_updatestate();
void evaluateCost_UpdateState();
void updateState(int,int,int,int);
void generateFeatureActionSpace();
void simulate();

void updateDCost();
int generateData(int);
int harvestEnergy();
int generatePoissonSample(int);
void saveResult();

// State-action aggregated space specific functions
void setupPartitionMap();
void generateFeatureActionSpace();

// ql specific functions
void getActualSplit();
void QLearning();
void getMinAction();
void savePolicy();
void suggestAction();
void prevState_QUpdate();
void resetDataStructures();

// cnql specific functions
void getSplit();
void cnql_savepolicy();

//epsilon-greedy and ucb specific functions
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
	cout << "lambda_x1 = "<<lambda_x1 << ", lambda_x2 = " <<lambda_x2 << 
	", lambda_x3 = " <<lambda_x3 << ", lambda_x4 = " <<lambda_x4 << endl;
	cout << "lambda_y = "<<lambda_y <<endl;
	cout << "DMAX = "<< DMAX<< ", EMAX = "<< EMAX << endl;
	cout << "Max data arrival: "<< Max_Data << endl;
	cout << "Max energy arrival: " << Max_Energy << endl;
	setupPartitionMap();
	generateFeatureActionSpace();

	    switch(algo)
	     {
		
		case 2: //combined-node ql
			resultsFile.open("cnql-dcost-4n-mdp.txt",ofstream::out|ofstream::app);
			policyFile.open("cnql-policy-4n-mdp.txt",ofstream::out);
			cout << "Combined Nodes method"<< endl;
			QLearning();
			break;
		
		case 3: //epsilon-greedy ql
		        resultsFile.open("qle-avgcost-4n-mdp.txt",ofstream::out|ofstream::app);
		        policyFile.open("qle-policy-4n-mdp.txt",ofstream::out);	
			cout << "Epsilon greedy Q-Learning"<< endl;		
			QLearning();
			//resetDataStructures();
			break;
		case 4: //ucb ql
			resultsFile.open("qlucb-cost-4n-mdp.txt",ofstream::out|ofstream::app);
			policyFile.open("qlucb-policy-4n-mdp.txt",ofstream::out);
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
	dCost = 0.0;
	reward = 0.0;
	qk1 = 0 ; qk2 = 0 ; qk3 = 0 ; qk4 = 0 ; E = 0 ;
	a1 = 0; a2 = 0; a3 = 0 ; a4 = 0 ;
	qk1P = 0, qk2P = 0, qk3P = 0, qk4P = 0, EP = 0;
	agA1 = 0 ; agA2 = 0 ; agA3 = 0; agA4 = 0;	
	agAP1 = 0; agAP2 = 0; agAP3 = 0; agAP4 = 0;
	
	// Stores the aggregate state
	as1 = 0 ; as2 = 0 ; as3 = 0 ; as4 = 0 ; as5 = 0;
	asP1 = 0 ; asP2 = 0 ; asP3 = 0 ; asP4 = 0 ; asP5 = 0;

	nStep = 0;
	x1P = x2P = x3P = x4P = 0;
	eYP = 0;
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

	nStep = 0;

	dCost = 0.0; reward = 0.0;
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
	  case 2: if (!freezeLearning)
	   	   {	    
	                learnCost_updatestate();
	                prevState_QUpdate();
	       		suggestAction();	  
	       		cnV[qk1+qk2+qk3+qk4][E][x1][x2][x3][x4][eY][A] = true;
	   	   }
		   else
	   	   {
	       		nStep++;
	       		evaluateCost_UpdateState();
	       		updateDCost();
	       		A = cnP[qk1+qk2+qk3+qk4][E][x1][x2][x3][x4][eY];
	       		getSplit();			
	   	   }	
		   break;
	   
	   case 3: case 4:
		   if (!freezeLearning)
		   {
			learnCost_updatestate();
			as1 = dfsMap.at(qk1); as2 = dfsMap.at(qk2); 
			as3 = dfsMap.at(qk3); as4 = dfsMap.at(qk4); 
			as5 = fsMap.at(E);

	                prevState_QUpdate();
	       		suggestAction();
		 	vcount[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY]++;
		 isvisited[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][agA1][agA2][agA3][agA4] = true;

		        if (algo == 4) 
			    savcount[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][agA1][agA2][agA3][agA4]++;
		   }
		   else
		   {
	       		nStep++;
	       		evaluateCost_UpdateState();
			as1 = dfsMap.at(qk1); as2 = dfsMap.at(qk2); 
			as3 = dfsMap.at(qk3); as4 = dfsMap.at(qk4); 
			as5 = fsMap.at(E);

	       		updateDCost();
	                agA1 = policy[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][0];
	       		agA2 = policy[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][1];
			agA3 = policy[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][2];
	       		agA4 = policy[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][3];
			getActualSplit();
		   }
		   break;
		
	}
	qk1P = qk1;
	qk2P = qk2;
	qk3P = qk3;
	qk4P = qk4;
	EP = E;

	x1P = x1;
	x2P = x2;
	x3P = x3;
	x4P = x4;
	eYP = eY;

	agAP1 = agA1;	// low,medium or high action
	agAP2 = agA2;
	agAP3 = agA3;
	agAP4 = agA4;

}


void prevState_QUpdate()
{
	float tmp = 0.0,tmp1 = 0.0;
	getMinAction();
	
	switch (algo)
	{
	 case 2:{
		int st = qk1+qk2+qk3+qk4;
	   	int stP = qk1P+qk2P+qk3P+qk4P;

		tmp = cnQ[stP][EP][x1P][x2P][x3P][x4P][eYP][A];
		tmp1 = cnQ[st][E][x1][x2][x3][x4][eY][minActionIndex];
		tmp = (float)((1-alpha_stepsize)*tmp + alpha_stepsize*(reward + discountF*tmp1));			
		cnQ[stP][EP][x1P][x2P][x3P][x4P][eYP][A] = tmp;
		}
		break;
	case 3: case 4:
		{
		vector<int> u = acList[minActionIndex];
		int minA1 = u[0]; int minA2 = u[1]; int minA3 = u[2]; int minA4 = u[3];		

		tmp = Q[asP1][asP2][asP3][asP4][asP5][x1P][x2P][x3P][x4P][eYP][agAP1][agAP2][agAP3][agAP4];
		tmp1 = Q[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][minA1][minA2][minA3][minA4];

		tmp = (float)((1-alpha_stepsize)*tmp + alpha_stepsize*(reward + discountF*tmp1));
		Q[asP1][asP2][asP3][asP4][asP5][x1P][x2P][x3P][x4P][eYP][agAP1][agAP2][agAP3][agAP4] = tmp;
		}
		break;
	}
}


void getMinAction()
{		
	switch (algo)
	{
	   case 2:{
		   int st = qk1+qk2+qk3+qk4;
		  minActionIndex = 0;
	   	  for (int i = 1 ; i <= E ; i++)
	   	  {	  
	       	     if (cnQ[st][E][x1][x2][x3][x4][eY][i] < cnQ[st][E][x1][x2][x3][x4][eY][minActionIndex])
	   	  	minActionIndex = i;
	   	  }
		  }
		  break;
	   case 3:
		  acList = fHmap.at(as5);
		  minActionIndex = 0;
		  for (vector<int>::size_type i=1 ; i< acList.size() ; i++)
		  {
		      vector<int> y = acList[minActionIndex];
	  	      int mA1 = y[0]; int mA2 = y[1]; int mA3 = y[2]; int mA4 = y[3];

	  	      vector<int> x = acList[i];
	  	      int cA1 = x[0]; int cA2 = x[1]; int cA3 = x[2]; int cA4 = x[3];

		      if (Q[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][cA1][cA2][cA3][cA4] < 
			  Q[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][mA1][mA2][mA3][mA4])
				minActionIndex = i;
		  }
		  break; 
	   case 4:
		  acList = fHmap.at(as5);
		  minActionIndex = 0;
		  for (vector<int>::size_type i=1 ; i< acList.size() ; i++)
		  {
		      vector<int> y = acList[minActionIndex];
	  	      int mA1 = y[0]; int mA2 = y[1]; int mA3 = y[2]; int mA4 = y[3];

	  	      vector<int> x = acList[i];
	  	      int cA1 = x[0]; int cA2 = x[1]; int cA3 = x[2]; int cA4 = x[3];
		
		      int svc = vcount[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY];
	  	      int savc1 = savcount[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][mA1][mA2][mA3][mA4];
	  	      int savc2 = savcount[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][cA1][cA2][cA3][cA4];

		      double val1 = 0.0,val2 = 0.0;
	  	      if (savc1 > 0)  // max{ -Q(s,a) + sqrt(ln(N(s))/N(s,a)) }	    
	     	          val1 = alpha_ucb*( sqrt( log(svc)/savc1 ) ) - 
				Q[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][mA1][mA2][mA3][mA4] ;
	  	      else val1 = 1;

	  	      if (savc2 > 0)  
	                  val2 = alpha_ucb*( sqrt( log(svc)/savc2 ) ) - 
				Q[as1][as2][as3][as4][as5][x1][x2][x3][x4][eY][cA1][cA2][cA3][cA4];
	  	      else val2 = 1;

		      if (val2 > val1)
	   	    	minActionIndex = i;
		  }
		  break;
	}
}


int harvestEnergy()
{
	int y = (int)floor(c1*eYP + generatePoissonSample(5));
	return y;
}


void updateDCost()
{  dCost = dCost + (reward - dCost)/(nStep+1); }


void evaluateCost_UpdateState()
{
	double gk1 = convertEnergytoData(1);
	double gk2 = convertEnergytoData(2);
	double gk3 = convertEnergytoData(3);
	double gk4 = convertEnergytoData(4);

	int Gk1 = (int)floor(gk1);
	int Gk2 = (int)floor(gk2);
	int Gk3 = (int)floor(gk3);
	int Gk4 = (int)floor(gk4);

	double costNode1 = fmax((qk1 - Gk1),0);
	double costNode2 = fmax((qk2 - Gk2),0);
	double costNode3 = fmax((qk3 - Gk3),0);
	double costNode4 = fmax((qk4 - Gk4),0);

	reward = (costNode1 + costNode2 + costNode3 + costNode4)/DMAX;
	updateState(Gk1,Gk2,Gk3,Gk4);
}


void learnCost_updatestate()
{
	double gk1 = convertEnergytoData(1);
	double gk2 = convertEnergytoData(2);
	double gk3 = convertEnergytoData(3);
	double gk4 = convertEnergytoData(4);

	int Gk1 = (int)floor(gk1);
	int Gk2 = (int)floor(gk2);
	int Gk3 = (int)floor(gk3);
	int Gk4 = (int)floor(gk4);

	// Precision Error in splitting energy -- for (1,2,x) action (2,y) is better than (3,z)
	double err1 = gk1-Gk1;
	double err2 = gk2-Gk2;	 
	double err3 = gk3-Gk3;	    
	double err4 = gk4-Gk4;	       

	// Find error in the total sum of energy given (wrt Q-Learning on combined nodes)
	// If energy given to a node is more than the required energy 
	double err5 = 0.0,err6 = 0.0, err7 = 0.0,err8 = 0.0;
	if (Gk1 > qk1)
  	    err5 = Gk1-qk1;
	if (Gk2 > qk2)
	    err6 = Gk2-qk2;	    
	if (Gk3 > qk3)
  	    err7 = Gk3-qk3;
	if (Gk4 > qk4)
	    err8 = Gk4-qk4;	    

	double costNode1 = fmax((qk1 - Gk1),0) + err1 + err5;
	double costNode2 = fmax((qk2 - Gk2),0) + err2 + err6;
	double costNode3 = fmax((qk3 - Gk3),0) + err3 + err7;
	double costNode4 = fmax((qk4 - Gk4),0) + err4 + err8;

	reward = (costNode1 + costNode2 + costNode3 + costNode4)/DMAX;
	updateState(Gk1,Gk2,Gk3,Gk4);
}


int generateData(int node)
{
// Includes correlation of data arrival
	int x = 0;
	switch(node)
	{
	   case 1: x = (int)floor(b1*x1P + b2*x2P + b3*x3P + b4*x4P + generatePoissonSample(1));
		break;
	   case 2: x = (int)floor(d1*x1P + d2*x2P + d3*x3P + d4*x4P + generatePoissonSample(2));
		break;
	   case 3: x = (int)floor(f1*x1P + f2*x2P + f3*x3P + f4*x4P + generatePoissonSample(3));
		break;
	   case 4: x = (int)floor(h1*x1P + h2*x2P + h3*x3P + h4*x4P + generatePoissonSample(4));
		break;
	}
	return x;
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
	  case 3: lambda = lambda_x3;
		  break;
   	  case 4: lambda = lambda_x4;
		  break;
	  case 5: lambda = lambda_y;
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

void updateState(int Gk1,int Gk2,int Gk3,int Gk4)
{
	int x_k1 = generateData(1);
	int x_k2 = generateData(2);
	int x_k3 = generateData(3);
	int x_k4 = generateData(4);

	if (x_k1 > Max_Data) x_k1 = Max_Data;
	if (x_k2 > Max_Data) x_k2 = Max_Data;
	if (x_k3 > Max_Data) x_k3 = Max_Data;
	if (x_k4 > Max_Data) x_k4 = Max_Data;	

	int yk = harvestEnergy();
	if (yk > Max_Energy) yk = Max_Energy;

	E -= (a1+a2+a3+a4);	

	if (E < 0)
	    E = 0;
	E += yk;
	if (E > EMAX)
	    E = EMAX;

	// Update data queue length
	qk1 = fmax((qk1 - Gk1),0) + x_k1;
	if (qk1 > DMAX)	    	
	    qk1 = DMAX;

	qk2 = fmax((qk2 - Gk2),0) + x_k2;
	if (qk2 > DMAX)
	    qk2 = DMAX;

	qk3 = fmax((qk3 - Gk3),0) + x_k3;
	if (qk3 > DMAX)	    	
	    qk3 = DMAX;

	qk4 = fmax((qk4 - Gk4),0) + x_k4;
	if (qk4 > DMAX)
	    qk4 = DMAX;

	x1 = x_k1;
        x2 = x_k2;
	x3 = x_k3;
        x4 = x_k4;

	eY = yk;
}

double convertEnergytoData(int node)
{
	double dataSent = 0;
	switch(node)
	{
	   case 1: dataSent = 2*log(1+a1); break;
	   case 2: dataSent = 2*log(1+a2); break;
	   case 3: dataSent = 2*log(1+a3); break;
	   case 4: dataSent = 2*log(1+a4); break;
	}
	return dataSent;
}

void setupPartitionMap()
{
	for (int i = 0 ; i <= EMAX ; i++)
	{
	    if (i <= EL1up)
		fsMap.insert(make_pair(i,0));			
	    else if (i <= EL2up)
		fsMap.insert(make_pair(i,1));
	    else if (i <= EL3up)
		fsMap.insert(make_pair(i,2));	
	}

	for (int j = 0; j <= DMAX ; j++)
	{
	    if (j <= DL1up)
		dfsMap.insert(make_pair(j,0));
	    else if (j <= DL2up)
		dfsMap.insert(make_pair(j,1));
	}
}

void generateFeatureActionSpace()
{      
	// low energy has only low actions
	vector<vector<int> > t1,t2,t3;

	vector<int> lr1(4,0);	
	t1.push_back(lr1); t2.push_back(lr1);  t3.push_back(lr1);
	fHmap.insert(make_pair(0,t1));
// ------------------------------------------------------------------------------------------------------------	
	// med energy has only (L,L,L,M1),(L,L,M1,L),(L,M1,L,L),(M1,L,L,L) and (L,L,L,L)actions
	vector<int> er2(4,0),er3(4,0),er4(4,0),er5(4,0);

	er2[0] = 1; er3[1] = 1;	er4[2] = 1; er5[3] = 1;	

	t2.push_back(er2); t3.push_back(er2);
	t2.push_back(er3); t3.push_back(er3);
	t2.push_back(er4); t3.push_back(er4);
	t2.push_back(er5); t3.push_back(er5);	
	
	fHmap.insert(make_pair(1,t2));	
// ------------------------------------------------------------------------------------------------------------		
	vector<int> m5(4,0),m6(4,0),m7(4,0),m8(4,0),m9(4,0),m10(4,0);
	vector<int> high9(4,0),high10(4,0),high11(4,0),high12(4,0);

	//med2 energy has (L,L,M1,M1),(L,M1,L,M1),(M1,L,L,M1),(L,M1,M1,L),(M1,L,M1,L),(M1,M1,L,L) actions
	m5[2] = 1 ; m5[3] = 1 ;
	m6[1] = 1 ; m6[3] = 1 ;
	m7[0] = 1 ; m7[3] = 1 ;
	m8[1] = 1 ; m8[2] = 1 ;
	m9[0] = 1 ; m9[2] = 1 ;
	m10[0] = 1 ; m10[1] = 1 ;

	high9[3] = 2 ; high10[2] = 2 ; high11[1] = 2 ;	high12[0] = 2 ;
	
	t3.push_back(m5); t3.push_back(m6); t3.push_back(m7); t3.push_back(m8);
	t3.push_back(m9); t3.push_back(m10); 

	t3.push_back(high9); t3.push_back(high10); t3.push_back(high11); t3.push_back(high12);

	fHmap.insert(make_pair(2,t3));	
}


void getActualSplit()
{
	int lim = E;
	int sum1 = 0, sum2 = 0,sum3 = 0,sum4 = 0;
	int lim1 = 0, lim2 = 0, lim3 = 0, lim4 = 0;
       
     	if (E < 5) { a1 = 0; a2 = 0; a3 = 0; a4 = 0; return;}    
	switch (agA1)
	{
	    case 0: sum1 = EL1low; lim1 = EL1up;
		    break;
	    case 1: sum1 = EL2low; lim1 = EL2up; 
		    break;
	    case 2: sum1 = EL3low; lim1 = EL3up;
		    break;
	}

	switch (agA2)
	{
	    case 0: sum2 = EL1low; lim2 = EL1up;
		    break;
	    case 1: sum2 = EL2low; lim2 = EL2up; 
		    break;
	    case 2: sum2 = EL3low; lim2 = EL3up;
		    break;
	}

	switch (agA3)
	{
	    case 0: sum3 = EL1low; lim3 = EL1up;
		    break;
	    case 1: sum3 = EL2low; lim3 = EL2up; 
		    break;
	    case 2: sum3 = EL3low; lim3 = EL3up;
		    break;
	}

	switch (agA4)
	{
	    case 0: sum4 = EL1low; lim4 = EL1up;
		    break;
	    case 1: sum4 = EL2low; lim4 = EL2up; 
		    break;
	    case 2: sum4 = EL3low; lim4 = EL3up;
		    break;
	}
	lim -= (sum1+sum2+sum3+sum4);
	
	vector<int> nodes;
	nodes.push_back(1); nodes.push_back(2); nodes.push_back(3); nodes.push_back(4);
	vector<double> cdf;
	cdf.push_back(0.25);
	cdf.push_back(0.5);
	cdf.push_back(0.75);
	cdf.push_back(1.0);

	for (int k = 1 ; k <= lim && !nodes.empty(); k++)
	{
	   double pick = generator->drand();
	   int picked = 0;
	   for (vector<double>::size_type i = 0; i < cdf.size(); )
	   {		     
	     if (pick > cdf[i])
		 i++;
	     else
	     {
		picked = i;
		break;
	     }
	   }
	   switch(nodes[picked])
	    {
		case 1: sum1++;
			if (sum1 == lim1) nodes.erase(nodes.begin()+picked);
			break;
		case 2: sum2++;
			if (sum2 == lim2) nodes.erase(nodes.begin()+picked);
			break;
		case 3: sum3++;
			if (sum3 == lim3) nodes.erase(nodes.begin()+picked);
			break;
		case 4: sum4++;
			if (sum4 == lim4) nodes.erase(nodes.begin()+picked);
			break;
	    }
	   cdf.clear();
	   for (vector<int>::size_type it = 0 ; it < nodes.size(); it++)
		cdf.push_back( (it*1.0+1.0)/nodes.size() );
	}
	a1 = sum1 ; a2 = sum2 ; a3 = sum3 ; a4 = sum4;
}


void getSplit()// called by greedy and cnql methods
{
	if (algo == 1) A = E;	//if greedy, then use entire energy
	if (A > 0)
	{
	   vector<double> cdf;
	   int req1 = (int)ceil(exp(qk1/2.0) - 1);
	   int req2 = (int)ceil(exp(qk2/2.0) - 1);
	   int req3 = (int)ceil(exp(qk3/2.0) - 1);
	   int req4 = (int)ceil(exp(qk4/2.0) - 1);

	   int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
	
	   vector<int> nonFull;
	  
	   if (qk1 > 0 && !(sum1 == req1))								
		nonFull.push_back(1);
			
	   if (qk2 > 0 && !(sum2 == req2))								
		nonFull.push_back(2);

	   if (qk3 > 0 && !(sum3 == req3))								
		nonFull.push_back(3);
			
	   if (qk4 > 0 && !(sum4 == req4))								
		nonFull.push_back(4);
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
			case 3: denom += qk3;
				numerator += qk3;
				cdf.push_back(numerator);
				break;
			case 4: denom += qk4;
				numerator += qk4;
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
		    case 3:sum3++;
			   if (sum3 == req3) 
			      nonFull.erase(nonFull.begin()+picked);
			   break;
		    case 4:sum4++;
			   if (sum4 == req4)
			      nonFull.erase(nonFull.begin()+picked);
			   break;
		  }
	    }
	   a1 = sum1;
	   a2 = sum2;  
	   a3 = sum3;
	   a4 = sum4;
	}
	else
	{ a1 = 0; a2 = 0; a3 = 0; a4 = 0;}
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
		  agA1 = y[0]; agA2 = y[1]; agA3 = y[2]; agA4 = y[3];
		  getActualSplit();
		  break;
	   case 4:vector<int> u = acList[minActionIndex]; 
	   	  agA1 = y[0]; agA2 = y[1]; agA3 = y[2]; agA4 = y[3];
		  getActualSplit();
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
		   for (int xk3 = 0 ; xk3 <= Max_Data ; xk3++)
		   {
		      for (int xk4 = 0 ; xk4 <= Max_Data ; xk4++)
	   	      {
		         for (int yk = 0; yk <= Max_Energy ; yk++)
		         {		  
		             int j = 0;
	     	             for ( ; j <= energy && !cnV[state][energy][xk1][xk2][xk3][xk4][yk][j]; j++) ;
	     		     int ma = 0;
			     if (j <= energy) 
	     		     {
			        ma = j;
			        for (int k = ma+1; k <= energy ; k++)
			         {
				    if ( (cnQ[state][energy][xk1][xk2][xk3][xk4][yk][k] < 
					  cnQ[state][energy][xk1][xk2][xk3][xk4][yk][ma]) && 
					  cnV[state][energy][xk1][xk2][xk3][xk4][yk][k] )
		 	   		ma = k;
			   	 }
	     		     }
	     		     cnP[state][energy][xk1][xk2][xk3][xk4][yk] = ma;
	     		     policyFile << "\n State: " << state << ", "<< energy <<", "<< xk1 <<", " << xk2 <<
			     ", "<<xk3<<", "<<xk4<<", "<<yk << "\t Action: "<< ma ;
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
	resultsFile << "\n--------------------------------------------";
	resultsFile << "\nDMAX = " <<DMAX << ", EMAX = "<<EMAX << "\n";
	resultsFile << "lambda_x1 = " <<lambda_x1 << ", lambda_x2 = " <<lambda_x2 
	 << ", lambda_x3 = " <<lambda_x3 << ", lambda_x4 = " <<lambda_x4 ;
	resultsFile << "\nlambda_y = " <<lambda_y ;
	
	if (algo >= 2) resultsFile << "\n Epsilon = "<<epsilon;
	resultsFile << "\n Learning steps: " <<learnSteps;
	resultsFile << "\n Evaluation steps: " <<evalSteps;

	resultsFile <<"\n Average cost: "<< dCost ;
}


void eql_savepolicy()
{
  for (int en = 1 ; en < nlE ; en++) // for level 0 the only action is (l,l,l,l)
  {
    acList = fHmap.at(en);	// get me the available action levels
		
    for (int q1 = 0; q1 < nlD ; q1++)
    {
     for (int q2 = 0; q2 < nlD ; q2++)
     {
      for (int q3 = 0; q3 < nlD ; q3++)
      {
	for (int q4 = 0; q4 < nlD ; q4++)
	{
	  for (int xk1 = 0;xk1 <= Max_Data;xk1++)
	  {
	    for (int xk2 = 0; xk2 <= Max_Data; xk2++)
	    {
	      for (int xk3 = 0; xk3 <= Max_Data ; xk3++)
	      {
 		for (int xk4 = 0;xk4 <= Max_Data; xk4++)
		{
		  for (int yk = 0 ; yk<= Max_Energy ; yk++)
		  {
		   vector<int> indices;
		   for (vector<int>::size_type y = 0; y < acList.size(); y++)
		   {
		     vector<int> x = acList[y];
		     int cA1 = x[0]; int cA2 = x[1]; int cA3 = x[2]; int cA4 = x[3];
		     if (Q[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][cA1][cA2][cA3][cA4] > 0.0 
			 || isvisited[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][cA1][cA2][cA3][cA4])
			          indices.push_back(y);			
		   }
		  
		   minActionIndex = 0;
		   for (vector<int>::size_type u = 1; u < indices.size() ; u++)
		   {
		    int indexM = indices[minActionIndex];
		    int indexC = indices[u];						
						
		    vector<int> u1 = acList[indexM];
		    int mA1 = u1[0]; int mA2 = u1[1];
		    int mA3 = u1[2]; int mA4 = u1[3];
						
		    vector<int> u2 = acList[indexC];
		    int cA1 = u2[0]; int cA2 = u2[1];
		    int cA3 = u2[2]; int cA4 = u2[3];
						
	            if (Q[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][cA1][cA2][cA3][cA4] < 
			Q[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][mA1][mA2][mA3][mA4])
			minActionIndex = u;			
		   }
		   if (indices.size() > 0)
		   {	
		      int indexM = indices[minActionIndex];	
		      vector<int> z = acList[indexM];	 
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][0] = z[0];
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][1] = z[1];
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][2] = z[2];
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][3] = z[3];

		      policyFile << "\nState: " << q1 << "," << q2 <<"," << q3 <<"," << q4 <<"," << en
				<<", "<< xk1<<", "<< xk2<<", "<< xk3<<", "<< xk4<<", "<< yk			       
				<< "\tAction: "<< z[0] << "," << z[1] <<"," << z[2] << "," << z[3];
		   }
		   else 
		   {
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][0] = 0;
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][1] = 0;
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][2] = 0;
		      policy[q1][q2][q3][q4][en][xk1][xk2][xk3][xk4][yk][3] = 0;

    		      policyFile << "\nState: " << q1 << "," << q2 <<"," << q3 <<"," << q4 <<"," << en
				<<", "<< xk1<<", "<< xk2<<", "<< xk3<<", "<< xk4<<", "<< yk <<"\tAction: 0,0,0,0";	        
		   }
		}
	     }
	   }
	  }
	}
       }
      }
     }
    }
   }
}
