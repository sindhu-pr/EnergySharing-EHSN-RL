#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>
#include "randgen.h"

using namespace std;

float b1 = 0.1, b2 = 0.1, b3 = 0.1, b4 = 0.2;  // 1st node data generation co-eff
float d1 = 0.1, d2 = 0.1, d3 = 0.2, d4 = 0.1;  // 2nd node data generation co-eff
float f1 = 0.1, f2 = 0.2, f3 = 0.1, f4 = 0.1;  // 
float h1 = 0.2, h2 = 0.1, h3 = 0.1, h4 = 0.1;  // 
float c1 = 0.5;   // energy

// Buffer sizes
const int EMAX = 25,DMAX = 10;
const int baseD1 = DMAX+1;
const int baseE = EMAX+1;
const int baseD2 = 4*DMAX + 1;

// Maximum value of energy arrival
const int Max_Energy = 15;
// Maximum value of data arrival
const int Max_Data = 4;

// number of partitions of data and energy buffers
const int nlE = 4;
const int nlD = 2;

//manual partitions - Energy and data threshold levels
int DL1low = 0, DL1up = 5;
int DL2low = 6, DL2up = 10;

int EL1low = 0, EL1up = 7;
int EL2low = 8, EL2up = 15;
int EL3low = 16, EL3up = 20;
int EL4low = 21,EL4up = 25;

// Cost variables
double reward = 0.0;
double dCost = 0.0;

//Rates for noise variable for data and energy generation
double lambda_x1 = 1,lambda_x2 = 1, lambda_x3 = 1, lambda_x4 = 1;
double lambda_y = 5;

// Number of steps taken
long nStep = 0;

// Actual state variables
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

// Files to store the outputs - avgcost and policy
ofstream resultsFile,parametersFile;

// To learn or evaluate ?
bool freezeLearning = false;

// Number of iterations required to evaluate policy
long evalSteps = 20000000;

// partition-state map
map<int,int> fsMap,dfsMap;

// Mapping aggregate states to aggregate actions available
map<int,vector<vector<int> > > fHmap;
vector< vector<int> > acList;

// random generator
randgen* generator;

//Quantile value
const double quantile = 0.4;

// Keeps track of average cost per 
// step of every sample trajectory
double rho = 0.0;

// Number of samples used in theta updation
const int N = 400;

// Number of components in theta updation
const int nComponents = 200;
const int nThetaSamples = 1000;

const int nUpd = 1000;	// 500 more than enough
const int samplePathSteps = 3000;

// Tracks sample number in policy search
int sampleNum = -1;

//Temperature Parameter for Policy
double tau = 1.0;

// Data structures for storing features
vector< vector<double> > phiList;

// Data structure for storing policy
vector<double> piSpAp;

// Data structures for storing rho values of
// sample trajectories
map<double,int> rhoMap;
vector<double> rhoVector;

// Data structures used for storing theta vectors
double thetaMat[nThetaSamples][nComponents];
double theta[nComponents];
double meanVector[nComponents],varianceVector[nComponents],indices[N];

// Function declarations
double convertEnergytoData(int);
void learnCost_updatestate();
void evaluateCost_UpdateState();
void updateState(int,int,int,int);
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

// Cross entropy specific functions
void Cross_Entropy();
void fixMeanValues();
void fixVariance();
void pickNextTheta();
void pickBestTheta();
void updDistro_Parameters();
void updateRho();
void computePi();
vector<double> getPhi(int,int,int,int,int,int,int,int,int,int,int,int,int,int);
void setNextAction();
void evalPickTheta();
void printMean();
void printVariance();

int main(int argc,char*argv[])
{
	generator = new randgen();
	if (argc < 2)
	{
	    cout << "\nUsage: *.out <Node1 noise rate>\n" << endl;
	    exit(0);
	}
	lambda_x1 = atof(argv[1]);
	cout << "lambda_x1 = "<<lambda_x1 << ", lambda_x2 = " <<lambda_x2 << 
	", lambda_x3 = " <<lambda_x3 << ", lambda_x4 = " <<lambda_x4 << endl;
	cout << "lambda_y = "<<lambda_y <<endl;
	cout << "DMAX = "<< DMAX<< ", EMAX = "<< EMAX << endl;
	cout << "Max data arrival: "<< Max_Data << endl;
	cout << "Max energy arrival: " << Max_Energy << endl;

	setupPartitionMap();	
	generateFeatureActionSpace();
	resultsFile.open("ce-dcost-mdp-change.txt",ofstream::out|ofstream::app);	
	parametersFile.open("ce-params-mdp-change.txt",ofstream::out);

	cout << "Cross Entropy Method"<< endl;
	Cross_Entropy();
	
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
	rho = 0.0;

	x1P = x2P = x3P = x4P = 0;
	eYP = 0;
}


void Cross_Entropy()
{
	fixMeanValues();
	fixVariance();
	parametersFile << "Initial Mean\n\n";
	printMean();
	parametersFile <<"\nInitial Variance\n\n";
	printVariance();

	init_data();	

	// Search policy
	freezeLearning = false;
	srand(time(0));

	
	for (int i= 1 ; i <= nUpd; i++)
	{
	   if (i%100 == 0) cout << "Updation "<<i<<endl;
	   for (int j = 1 ; j <= nThetaSamples ; j++)
	   {
		sampleNum++;
		pickNextTheta();
		init_data();
		
	  	for (int steps = 0 ; steps < samplePathSteps ; steps++)  
		     simulate(); 

		rhoVector.push_back(rho);
		rhoMap.insert(make_pair(rho,sampleNum));
	   }
	   pickBestTheta();
	   updDistro_Parameters();
	   sampleNum = -1;		
	   rhoMap.clear();
	   rhoVector.clear();
	   rho = 0.0;
	   nStep = 0;
	}
	parametersFile << "\nMean after all updations \n\n";
	printMean();
	parametersFile <<"\nVariance after all updations \n\n";
	printVariance();
	
	// Policy evaluation
	init_data();
	cout << "Policy Evaluation" <<endl;
	freezeLearning = true;
	evalPickTheta();

	srand(time(0));
        for (long i = 1 ; i <= evalSteps ; i++) 
	   { simulate(); }

	saveResult();
	resultsFile.close();	
	parametersFile.close();
}

void simulate()
{
	nStep++;

	if (!freezeLearning)
	{	    
	   learnCost_updatestate();	
	   updateRho();
	}
	else
	{ 	   
	   evaluateCost_UpdateState();
	   updateDCost();	   
	}
	as1 = dfsMap.at(qk1); as2 = dfsMap.at(qk2); 
	as3 = dfsMap.at(qk3); as4 = dfsMap.at(qk4); 
	as5 = fsMap.at(E);
	
	computePi();
	setNextAction();

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

int harvestEnergy()
{
	int y = (int)floor(c1*eYP + generatePoissonSample(5));
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

void updateRho()
{ rho = rho + (reward - rho)/(nStep+1);	}

void updateDCost()
{  dCost = dCost + (reward - dCost)/(nStep+1); }


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

void computePi()
{	
	piSpAp.clear();	

	acList = fHmap.at(as5);
	
	vector<double> ePower(acList.size(),0.0);
	double sumEPowers = 0.0;
	
	phiList.clear();
	for (unsigned int i=0 ; i< acList.size() ; i++)
	{
	    vector<int> ilist = acList[i];
	    int xa1 = ilist[0];
   	    int xa2 = ilist[1];
	    int xa3 = ilist[2];
   	    int xa4 = ilist[3];
	    vector<double> featuresB = getPhi(as1,as2,as3,as4,as5,x1,x2,x3,x4,eY,xa1,xa2,xa3,xa4);
	    phiList.push_back(featuresB);
	     
	    double prod = 0.0;
	    for (int k = 0; k< nComponents ; k++)		
	         prod += theta[i]*featuresB[i];
	    ePower[i] = exp( prod/tau );
	    sumEPowers = sumEPowers + ePower[i];	
	}
	  
	for (unsigned int j = 0 ; j < acList.size() ; j++)
	    piSpAp.push_back(ePower[j]/sumEPowers);
}


vector<double> getPhi(int x1,int x2,int x3,int x4,int x5,int x6,int x7,int x8,int x9,int x10,int y1,int y2,int y3,int y4)
{
	vector<double> features(nComponents,0.0);
	
	int dbase = nlD;
	int ebase = nlE;
	double state_action = 
	x1*pow(dbase,13)+x2*pow(dbase,12)+x3*pow(dbase,11)+x4*pow(dbase,10)+x5*pow(ebase,9)+ 
	x6*pow(Max_Data+1,8)+x7*pow(Max_Data+1,7) + x8*pow(Max_Data+1,6) + x9*pow(Max_Data+1,5) + x10*pow(Max_Energy+1,4)+
	y1*pow(ebase,3)+y2*pow(ebase,2)+y3*ebase+y4;

	// even - cosine functions, odd -sine functions
	for (int x = 0 ; x < nComponents ; x++)
	{
	    if ((x+1)%2 == 0)
	     { 
		double quo = ceil((x+1)/2);
		features[x] = cos(quo*state_action);	// cos xi
	     }
	     else
	     {
		double quo = ceil((x+1)/2);
		features[x] = sin(quo*state_action);
	     }			
	}
	return features;
}

void setNextAction()
{
	double cdf = 0.0;
	int actionIndex = 0;
	double pick = generator->drand();
	for (vector<double>::size_type i = 0; i < acList.size();  )
	{
	    cdf += piSpAp[i];
	    if (pick > cdf) i++;
	    else {
		actionIndex = i;
		break;
		}
	}
	vector<int> iList = acList.at(actionIndex);
	agA1 = iList[0];
	agA2 = iList[1];
	agA3 = iList[2];
	agA4 = iList[3];
	
	getActualSplit();
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
	    case 3: sum1 = EL4low; lim1 = EL4up;
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
	    case 3: sum2 = EL4low; lim2 = EL4up;
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
	    case 3: sum3 = EL4low; lim3 = EL4up;
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
	    case 3: sum4 = EL4low; lim4 = EL4up;
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


void saveResult()
{
	resultsFile << "\n--------------------------------------------";
	resultsFile << "\nDMAX = " <<DMAX << ", EMAX = "<<EMAX << "\n";
	resultsFile << "lambda_x1 = " <<lambda_x1 << ", lambda_x2 = " <<lambda_x2 
	 << ", lambda_x3 = " <<lambda_x3 << ", lambda_x4 = " <<lambda_x4 ;
	resultsFile << "\nlambda_y = " <<lambda_y ;
	resultsFile << "\nTheta components: "<< nComponents;
	resultsFile << "\nNo. of updations: "<< nUpd;
	resultsFile <<"\nNo. of sample path steps: "<< samplePathSteps;
	resultsFile <<"\n Average cost: "<< dCost ;
}

void fixMeanValues()
{
	srand(time(0));
	for (int i = 0 ; i < nComponents ; i++)
	     meanVector[i] = generator->drand();
}

void fixVariance()
{
	for (int i = 0 ; i < nComponents ; i++)
   	    varianceVector[i] = 1.0;
}


void pickNextTheta()
{
	srand(time(0));
	for (int i = 0 ; i< nComponents ; i++)
	{
             theta[i] = (generator->random_normal())*sqrt(varianceVector[i]) + meanVector[i];
	     thetaMat[sampleNum][i] = theta[i];
	}
}

void pickBestTheta()
{
	sort(rhoVector.begin(),rhoVector.end());
	reverse(rhoVector.begin(), rhoVector.end()); //sorted in descending order
	int fromIndex = (1-quantile)*nThetaSamples;
	for (int k = fromIndex, i = 0; k < nThetaSamples && i < N; k++,i++)
	     indices[i] = rhoMap.at(rhoVector[k]);
	/*
	resultsFile <<"RHO-QUANTILE = " << std::to_string(rhoVector[fromIndex]),3);
	resultsFile <<"Least Rho = "<< std::to_string(rhoVector[nThetaSamples-1]),3);
	cout << "RHO-QUANTILE = " << rhoVector[fromIndex] << endl;
	cout << "Least Rho = " << rhoVector[nThetaSamples-1] << "\n" <<endl;
	*/
}

void updDistro_Parameters()
{
	static int nu = 0;
	
	double updMeanVect[nComponents] = {0.0};	// how to fill with zeros
	double updVarVect[nComponents] = {0.0};
		
	for (int k = 0 ; k < N ; k++)
	{	     
	     unsigned int index = indices[k];
	     for (int x = 0 ; x < nComponents ; x++)
		updMeanVect[x] += thetaMat[index][x];		
	}
	for (int x = 0 ; x < nComponents ; x++)
	    meanVector[x] = updMeanVect[x]/N;
	
	for (int k = 0 ; k < N ; k++)
	{	    
	     unsigned int index = indices[k];
	     for (int x = 0 ; x < nComponents ; x++)
		updVarVect[x] += pow((thetaMat[index][x]-meanVector[x]),2);	
	}
	for (int x = 0 ; x < nComponents ; x++)
	    varianceVector[x] = updVarVect[x]/N;

	nu++;	
}


void evalPickTheta()
{
	for (int i = 0 ; i< nComponents ; i++)	
             theta[i] = meanVector[i];
}


void printMean()
{
	for (int v = 0; v < nComponents; v++)
   	    	parametersFile << std::setprecision(12) << meanVector[v] <<"\n";
	parametersFile.flush();
}

void printVariance()
{
	for (int v = 0; v < nComponents; v++)
   	    	parametersFile << std::setprecision(12) << varianceVector[v] <<"\n";
	parametersFile.flush();
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
	    else if (i <= EL4up)
		fsMap.insert(make_pair(i,3));
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
	vector<vector<int> > EL1Aset,EL2Aset,EL3Aset,EL4Aset;

	vector<int> allL1(4,0);	
	EL1Aset.push_back(allL1); EL2Aset.push_back(allL1);  
	EL3Aset.push_back(allL1); EL4Aset.push_back(allL1);

	fHmap.insert(make_pair(0,EL1Aset));
// ------------------------------------------------------------------------------------------------------------	
	vector<int> oneL2_1(4,0),oneL2_2(4,0),oneL2_3(4,0),oneL2_4(4,0);

	oneL2_1[0] = 1; oneL2_2[1] = 1;	oneL2_3[2] = 1; oneL2_4[3] = 1;	

	EL2Aset.push_back(oneL2_1); EL3Aset.push_back(oneL2_1); EL4Aset.push_back(oneL2_1);
	EL2Aset.push_back(oneL2_2); EL3Aset.push_back(oneL2_2); EL4Aset.push_back(oneL2_2);
	EL2Aset.push_back(oneL2_3); EL3Aset.push_back(oneL2_3); EL4Aset.push_back(oneL2_3);
	EL2Aset.push_back(oneL2_4); EL3Aset.push_back(oneL2_4); EL4Aset.push_back(oneL2_4);
	
	fHmap.insert(make_pair(1,EL2Aset));
//-------------------------------------------------------------------------------------------------
	vector<int> twoL2_1(4,0),twoL2_2(4,0),twoL2_3(4,0),twoL2_4(4,0),twoL2_5(4,0),twoL2_6(4,0);
	vector<int> oneL3_1(4,0),oneL3_2(4,0),oneL3_3(4,0),oneL3_4(4,0);

	twoL2_1[2] = 1 ; twoL2_1[3] = 1 ;
	twoL2_2[1] = 1 ; twoL2_2[3] = 1 ;
	twoL2_3[0] = 1 ; twoL2_3[3] = 1 ;
	twoL2_4[1] = 1 ; twoL2_4[2] = 1 ;
	twoL2_5[0] = 1 ; twoL2_5[2] = 1 ;
	twoL2_6[0] = 1 ; twoL2_6[1] = 1 ;

	oneL3_1[3] = 2 ; oneL3_2[2] = 2 ; oneL3_3[1] = 2 ;  oneL3_4[0] = 2 ;
	
	EL3Aset.push_back(twoL2_1);  EL3Aset.push_back(twoL2_2); EL3Aset.push_back(twoL2_3); 
	EL3Aset.push_back(twoL2_4);  EL3Aset.push_back(twoL2_5); EL3Aset.push_back(twoL2_6); 

	EL3Aset.push_back(oneL3_1); EL3Aset.push_back(oneL3_2); 
	EL3Aset.push_back(oneL3_3); EL3Aset.push_back(oneL3_4);

	EL4Aset.push_back(twoL2_1);  EL4Aset.push_back(twoL2_2); EL4Aset.push_back(twoL2_3); 
	EL4Aset.push_back(twoL2_4);  EL4Aset.push_back(twoL2_5); EL4Aset.push_back(twoL2_6); 

	EL4Aset.push_back(oneL3_1); EL4Aset.push_back(oneL3_2); 
	EL4Aset.push_back(oneL3_3); EL4Aset.push_back(oneL3_4);

	fHmap.insert(make_pair(2,EL3Aset));

//-------------------------------------------------------------------------------------------------
	vector<int> oneL4_1(4,0),oneL4_2(4,0),oneL4_3(4,0),oneL4_4(4,0);
	oneL4_1[0] = oneL4_2[1] = oneL4_3[2] = oneL4_4[3] = 3;

	EL4Aset.push_back(oneL4_1);  EL4Aset.push_back(oneL4_2);  
	EL4Aset.push_back(oneL4_3);  EL4Aset.push_back(oneL4_4);

	fHmap.insert(make_pair(3,EL4Aset));

}
