//================================================================================================
// Name        : RandomDataMatrix.cpp
// Author      : Katharina Jahn
// Version     : not the final one
// Copyright   : Your copyright notice
// Description : This creates simulated data sets. It creates
//					*	the true tree (.newick and .gv),
//					* 	the true data matrix,
//					*	the data matrix with noise
//					*	the data matrix with noise and missing data
//
//               The tree is rooted and has n+1 nodes (n mutations plus the wildtype root) which
//               is randomly drawn from the set of all trees of this type. If a double mutation is allowed
//				 there are n+2 nodes, w.l.o.g. the duplication is always for the node 1
//
//               The data matrix is a mxn data matrix (mutations x cells), for which a perfect phylogeny
//               exists that contains the tree.
//
//               Three types of noise are possible:
//               * value not observed (with probability na),
//               * false negative, 1 -> 0 (with probability fn), and
//               * false positive, 0 -> 1 (with probability fp).
//
//          the parameters read from the command line (fixed order):
//          <baseFileName> <n> <m> <fn rate> <fp rate> <na rate> <#trees to generate> <d>
//			The last parameter d is optional, if it is set a tree with one double mutation is created
//          e.g: test 100 150 0.1 1.24e-6 .001 2 d
//===============================================================================================


#include <string>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <locale>
#include <random>
#include <cstring>
#include "matrices.h"

using namespace std;

void initRand();
int* getRandTreeCode(int n);
int* prueferCode2parentVector(int* code, int codeLength);
bool** parentVector2ancMatrix(int* parent, int n);
vector<vector<int> > getChildListFromParentVector(int* parents, int n);
int** getRandomDataMatrix(int n, int m, bool** treeMatrix);
int** getRandomDataMatrixWithDoubleMut(int n, int m, bool** ancMatrix, int doubleMut);
//int** getNoisyMatrix(int** dataMatrix, int n, int m, double na);
int** getNoisyMatrix(int** dataMatrix, int n, int m, double fp, double fn);
int** addMissingValues(int** dataMatrix, int n, int m, double na);
string getNewickCode(vector<vector<int> > list, int root);
string getFileName(int i, string prefix, string params, string ending);
void writeToFile(string content, string fileName);
std::string getGraphVizFile2(int* parents, int n, int doubleMutation);
string matrixToString(int** matrix, int n, int m);
bool samplingByProb(double prob);
double sample_0_1();
int pickRandomNumber(int n);
int** cutLastRowFromMatrix(int** matrix, int n, int m);

bool* getInitialQueue(int* code, int codeLength);
int* getLastOcc(int* code, int codeLength);
int getNextInQueue(bool* queue, int pos, int length);
void updateQueue(int node, bool* queue, int next);
int updateQueueCutter(int node, bool* queue, int next);
int** pruneMatrix(int** oldMatrix, int n, int m);

bool ** transposeBoolMatrix(bool** matrix, int n, int m){
	//printf("transposing from %dx%d to %dx%d\n", n, m, m, n);
	bool ** transposed = allocate_boolMatrix(m, n);
	for(int i=0; i<m; i++){
		for(int j=0; j<n;j++){
			transposed[i][j] = matrix[j][i];
		}
	}
	return transposed;
}

bool addDoubleMut = false;

int main(int argc, char* argv[]) {

	initRand();

	string fileName = argv[1];             // base of file name
	int n = atoi(argv[2]);                 // number of mutations
	int m = atoi(argv[3]);                 // number of samples
	double fn = atof(argv[4]);             // false negative rate
	double fp = atof(argv[5]);             // false positive rate
	int na_min = atoi(argv[6]);            // min percentage of missing values
	int na_max = atoi(argv[7]);            // max percentage of missing values
	int rep = atoi(argv[8]);               // number of trees to be generated
	int maxDoubletCount = atoi(argv[9]);   // maximum number of doublets in data

	int doubleMut = -1;
	int nonRootNodeCount = n;
	int nodeCount = n+1;

	if(argc > 10 && strcmp(argv[10],"d")==0){   // generate trees with a double mutation
		addDoubleMut = true;
		nonRootNodeCount++;
		nodeCount++;
	}



	//cout << fileName << params.str() << "\n";
	for(int i=0; i< rep; i++){
		if(addDoubleMut){
			doubleMut = pickRandomNumber(n);      // pick a mutation to be the double mutation
		}
		stringstream params;
		params << "_n" << n << "_m" << m << fixed << setprecision(3) << "_fn" << fn << std::scientific << "_fp" << fp;   // string of parameters to add to file name
		if(addDoubleMut){
			params << "_d" << doubleMut;
		}

		int* treeCode = getRandTreeCode(nonRootNodeCount);
		int treeCodeLength = nonRootNodeCount-1;
		//print_intArray(treeCode, nodeCount-1);
		int* parentVector = prueferCode2parentVector(treeCode, treeCodeLength);
		//print_intArray(parentVector, nodeCount);
		vector<vector<int> > childLists = getChildListFromParentVector(parentVector, nonRootNodeCount);
		stringstream newick;
		newick << getNewickCode(childLists, nodeCount-1) << "\n";                                 // DFS of tree starting with id of root (n or n+1)
		//cout << newick.str();
		writeToFile(newick.str(), getFileName(i, fileName, params.str(), ".newick"));
		writeToFile(getGraphVizFile2(parentVector, nodeCount, doubleMut), getFileName(i, fileName, params.str(), ".gv"));
		//cout << getGraphVizFile2(parentVector, nodeCount, doubleMut);
		cout << getFileName(i, fileName, params.str(), ".gv") << "\n";
		bool** ancMatrix = parentVector2ancMatrix(parentVector, nonRootNodeCount);
		//print_boolMatrix(transposeBoolMatrix(ancMatrix, nodeCount, nodeCount), nodeCount, nodeCount);


		int** data;                                                                                   // for now this is the matrix where doublets are not yet merged
		if(doubleMut){                                                                                // the mutation states of the extra cells are at the end of the data matrix
			data = getRandomDataMatrixWithDoubleMut(n, m+maxDoubletCount, ancMatrix, doubleMut);       // therefore the data matrix has dimensions n x (m+maxDoubletCount)
		}
		else{
			data = getRandomDataMatrix(n,m+maxDoubletCount, ancMatrix);
		}
		//print_intMatrix(data, n, m+maxDoubletCount, ' ');
		//cout << "\n";




		// create doublet samples and add noise

		for(int x=0; x<maxDoubletCount; x++){    // for doublet x, merge mutation states of samples x and m+x to create a doublet

			for(int z=0; z<n; z++){
				//cout << "columns " << x << " and " << m+x << "\n";
				if(data[z][x] || data[z][m+x]){
					data[z][x] = true;
				}
				else{
					data[z][x] = false;               // state of mutation z is 1 if it is present in either sample x or sample m+x
				}
			}

			stringstream params2;
			params2 << params.str() << "_" << x+1 << "doublets";

			// add noise
			int** noisy =  getNoisyMatrix(data, n, m, fp, fn);

			//print_intMatrix(noisy, n, m, ' ');
			//cout << "\n";
			writeToFile(matrixToString(data,  n ,m), getFileName(i, fileName, params2.str(), ".data"));
			writeToFile(matrixToString(noisy, n, m), getFileName(i, fileName, params2.str(), ".noisy"));

			//cout << getFileName(i, fileName, params2.str(), ".data") << "\n";

			for(int j=na_min; j<=na_max; j++){
				double na_rate = j/100.0;
				//cout << "missing values rate: " << na_rate << "\n";
				int** missingData = addMissingValues(noisy, n, m, na_rate);
				stringstream ending;
				ending << ".noisy" << j << "pctMissingData";
				writeToFile(matrixToString(missingData, n, m), getFileName(i, fileName, params2.str(), ending.str()));
				//print_intMatrix(transposeMatrix(missingData, n, m), m, n, ' ');
				//cout << "\n";
				free_intMatrix(missingData);
			}
			free_intMatrix(noisy);
		}
		free_boolMatrix(ancMatrix);
		free_intMatrix(data);
		delete [] treeCode;
		delete [] parentVector;
	}
}

int** pruneMatrix(int** oldMatrix, int n, int m){
	int** matrix = allocate_intMatrix(n, m);

	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			//cout << "[" << i << "," << j << "]";
			matrix[i][j] = oldMatrix[i][j];
		}
	}
	free_intMatrix(oldMatrix);
	return matrix;
}

int** cutLastRowFromMatrix(int** matrix, int n, int m){
	int** smallerMatrix = init_intMatrix(n-1, m, 0);
	for(int i=0; i<n-1; i++){
		for(int j=0; j<m; j++){
			smallerMatrix[i][j] = matrix[i][j];
		}
	}
	return smallerMatrix;
}

string matrixToString(int** matrix, int n, int m){
	stringstream s;
	for(int i=0; i< n; i++){
		for(int j=0; j<m; j++){
			s << matrix[i][j];
			if(j!=m-1){
				s << " ";
			}
			else{
				s << "\n";
			}
		}
	}
	return s.str();
}

void writeToFile(string content, string fileName){
	ofstream outfile;
	outfile.open (fileName);
	outfile << content;
	outfile.close();
}

string getFileName(int i, string prefix, string params, string ending){
	stringstream fileName;
	fileName << prefix << "_" << i << "_" << params << ending;
	return fileName.str();
}

/* this function initializes the random number generator */
void initRand(){
	time_t t;
	time(&t);
	srand((unsigned int)t);
}


/* This function gets an integer n, and creates a random pruefer code for a rooted tree with n+1 nodes (root is always node n+1) */
int* getRandTreeCode(int n){                // n is the number of mutations

	int nodes = n+1;                        // #nodes = n mutations plus root
	int codeLength = nodes-2;
	int* code = new int[codeLength];
	//printf("getting random code of length %d now\n", codeLength);
	for(int i=0; i<codeLength; i++){
		code[i] = rand() % nodes;
		//cout << code[i] << " ";
	}
	//cout << "\n";
	return code;
}

/* given a Pruefer code, compute the corresponding parent vector */
int* prueferCode2parentVector(int* code, int codeLength){
	int nonRootNodeCount = codeLength+1;
	int rootId = nonRootNodeCount;
	int* parent = new int[nonRootNodeCount];

	int* lastOcc = getLastOcc(code, codeLength);    // node id -> index of last occ in code, -1 if no occurrence or if id=root
	bool* queue = getInitialQueue(code, codeLength);  // queue[node]=true if all children have been attached to this node, or if it is leaf
	int queueCutter = -1;    // this is used for a node that has been passed by the "queue" before all children have been attached
	int next = getNextInQueue(queue, 0, codeLength+1);

	for(int i=0; i<codeLength; i++){               // add new edge to tree from smallest node with all children attached to its parent
		if(queueCutter >=0){
			parent[queueCutter] = code[i];         // this node is queueCutter if the queue has already passed this node
			//cout << parent[queueCutter] << " -> " << queueCutter << "\n";
			queueCutter = -1;
		}
		else{
			parent[next] = code[i];                               // use the next smallest node in the queue, otherwise
			//cout << parent[next] << " -> " << next << "\n";
			next = getNextInQueue(queue, next+1, codeLength+1);     // find next smallest element in the queue

		}

		if(lastOcc[code[i]]==i){                               // an element is added to the queue, or we have a new queueCutter
			updateQueue(code[i], queue, next);
			queueCutter = updateQueueCutter(code[i], queue, next);
		}
	}
	if(queueCutter>=0){
		parent[queueCutter] = rootId;
		//cout << parent[queueCutter] << " -> " << queueCutter << "\n";
	}
	else{
		parent[next] = rootId;
		//cout << parent[next] << " -> " << next << "\n";
	}
	//printf("Parent vector: ");
	//print_intArray(parent, nodeCount);
	delete [] queue;
	delete [] lastOcc;
	return parent;
}

bool** parentVector2ancMatrix(int* parent, int n){
	bool** ancMatrix = init_boolMatrix(n, n, false);
	int root = n;
	for(int i=0; i<n; i++){
		int anc = i;
		int its =0;
		while(anc < root && its<40){                              // if the ancestor is the root node, it is not represented in the adjacency matrix
			//printf("%d -> %d\n", anc, parent[anc]);
			if(parent[anc]<n){
				ancMatrix[parent[anc]][i] = true;
			}

			anc = parent[anc];
			its++;
		}
	}
	for(int i=0; i<n; i++){
		ancMatrix[i][i] = true;
	}
	//print_boolMatrix(ancMatrix, n, n);
	return ancMatrix;
}

int** getRandomDataMatrix(int n, int m, bool** treeMatrix){
	int attachmentPoints = n+1;
	int** dataMatrix = init_intMatrix(n, m, 0);
	for(int i=0; i<m; i++){
		int parent = pickRandomNumber(attachmentPoints);
		if(parent==n){                                           // root node is parent, no entries need to be set (sample has no mutations)
			continue;
		}
		for(int j=0; j<n; j++){                             // dataMatrix: genes x samples
			dataMatrix[j][i] = treeMatrix[j][parent];       // treeMatrix[j][p] means j is an ancestor of p
		}
	}
	return dataMatrix;
}

int** getRandomDataMatrixWithDoubleMut(int n, int m, bool** ancMatrix, int doubleMut){
	int nodeCount = n+2;
	int doubleMutCopy = n;
	int** dataMatrix = init_intMatrix(n, m, 0);
	for(int sample=0; sample<m; sample++){
		int parent = pickRandomNumber(nodeCount);
		if(parent==nodeCount-1){                                 // parent is root node, no entries need to be set (sample has no mutations)
			continue;
		}

		for(int gene=0; gene<n; gene++){                             // set mutation status for each gene based on the attachment point
			dataMatrix[gene][sample] = ancMatrix[gene][parent];       // ancMatrix[j][p] means j is an ancestor of parent node

			if(gene==doubleMut && ancMatrix[doubleMut][parent] && ancMatrix[doubleMutCopy][parent]){   // back mutation, gene is not mutation in sample
				dataMatrix[gene][sample] = 0;
			}
		}
	}
	return dataMatrix;
}

/* This function gets a binary matrix and randomly changes the values according to missing value rate, false positive and false negative rates */
int** getNoisyMatrix(int** dataMatrix, int n, int m, double fp, double fn){
	std::default_random_engine generator;
	std::normal_distribution<double> norm(0.0, 0.1);
	// fix the error for the entire matrix
	double noisy_fn = fn * exp(norm(generator));      // add some noise
	int** noisyMatrix = deepCopy_intMatrix(dataMatrix, n, m);
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			                                        // then there is a chance the values are wrong:
			if(dataMatrix[i][j] == 0){                     // true value is wt
				if(samplingByProb(fp)){                    // perturb according to fp rate
					noisyMatrix[i][j] = 1;
				}
			}
			else{                                                 // true value is mutation
				//cout << noisy_fn << "\n";
				if(samplingByProb(noisy_fn)){                     // perturb according to noisy fn rate
					noisyMatrix[i][j] = 0;
				}
			}
		}
	}
	return noisyMatrix;
}

/* This function gets a binary matrix and randomly changes the values according to missing value rate, false positive and false negative rates */
int** addMissingValues(int** dataMatrix, int n, int m, double na){

	int** noisyMatrix = deepCopy_intMatrix(dataMatrix, n, m);
	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			if(samplingByProb(na)){                            // there is a chance this value is missing in the noisy matrix
				noisyMatrix[i][j] = 3;                         // missing values are encoded as 3
			}
		}
	}
	return noisyMatrix;
}

double getProbNormDistr(double prob, double stdDev){
	std::default_random_engine generator;
	std::normal_distribution<double> norm(0.0,stdDev);
	double y = norm(generator);
	double newProb = prob*exp(y);
	return newProb ;
}

/* converts a parent vector to the list of children */
vector<vector<int> > getChildListFromParentVector(int* parents, int n){
	int nodes = n+1;

	vector<vector<int> > childList(nodes);
	//cout << childList.size() << "\n";
	//for(int i=0; i<=n; i++){
	//	cout << childList.at(i).size() << "\n";
	//}
	for(int i=0; i<n; i++){
		//cout << "add " << i << " to " << parents[i] << "\n";
		childList.at(parents[i]).push_back(i);
	}
	//cout << "done\n";
	return childList;
}

/* converts a tree given as lists of children to the Newick tree format */
/* Note: This works only if the recursion is started with the root node which is n+1 */
string getNewickCode(vector<vector<int> > list, int root){
	stringstream newick;
	newick << root+1;
	vector<int> rootChilds = list.at(root);
	if(!rootChilds.empty()){
		newick << "(";
		bool first = true;
		for(int i=0; i<rootChilds.size(); i++){
			if(!first){
				newick << ",";
			}
			first = false;
			newick << getNewickCode(list, rootChilds.at(i));
		}
		newick << ")";
	}
	return newick.str();
}


/* samples a boolean value based on the probability for true (prob) */
bool samplingByProb(double prob){
	double randomNumber = sample_0_1();
	//double percent = rand() % 100;

	if(randomNumber <= prob){
		//cout << randomNumber << " <= " << prob << " ?\n";
		return true;
	}
	return false;
}



/* samples a random integer between 0 and n-1 */
int pickRandomNumber(int n){

    return (rand() % n);
}

double sample_0_1(){

  //return (((double) rand()+0.5) / ((RAND_MAX+1)));
  return ((double) rand() / RAND_MAX);
}


/* The next 5 functions are used for transforming the pruefer code into a parent vector */
bool* getInitialQueue(int* code, int codeLength){
	int queueLength = codeLength+1;
	int rootId = codeLength+1;
	bool* queue = init_boolArray(queueLength, true);
	for(int i=0; i<codeLength; i++){
		if(code[i]!=rootId){
			queue[code[i]] = false;
		}
	}
	return queue;
}

int* getLastOcc(int* code, int codeLength){
	int* lastOcc = init_intArray(codeLength+2, -1);
	int root = codeLength+1;
	for(int i=0; i<codeLength; i++){
		if(code[i] != root){
			lastOcc[code[i]] = i;
		}
	}
	return lastOcc;
}

int getNextInQueue(bool* queue, int pos, int length){
	for(int i=pos; i<length; i++){
		if(queue[i]==true){
			return i;
		}
	}
	//printf("No node left in queue. Possibly a cycle?");
	return length;
}

void updateQueue(int node, bool* queue, int next){

	if(node>=next){                //  add new node to queue
		queue[node] = true;
	}
}

int updateQueueCutter(int node, bool* queue, int next){
	//printf("node: %d, next in queue: %d\n", node, next);
	if(node>=next){
		return -1;         // new node can be added to the queue
	}
	else{
		return node;         // new node needs to cut the queue, as it has already passed it
	}
}

std::string getGraphVizFile2(int* parents, int nodeCount, int doubleMut){
	std::stringstream content;
	content << "digraph G {\n";
	content << "node [color=deeppink4, style=filled, fontcolor=white];\n";
	for(int i=0; i<nodeCount; i++){
		stringstream nodeLabel;
		stringstream parentLabel;
		if(i==nodeCount-2 && addDoubleMut){                                           // node i is second copy of double mutation
            cout << addDoubleMut << "\n";
			nodeLabel << "\"" << doubleMut+1 << "_copy" << "\"";
		}
		else{
			nodeLabel <<  i+1;                   // plus 1 to have gene labels start at 1
		}

		if(parents[i]==nodeCount-1){
			parentLabel << "\"root\"";
		}
		else if(parents[i]==nodeCount-2 && addDoubleMut){                                // parent of node i is second copy of double mutation
			parentLabel << "\"" <<doubleMut+1 << "_copy" << "\"";
		}
		else{
			parentLabel << parents[i]+1;
		}

		content << parentLabel.str();
		content << " -> ";
		content << nodeLabel.str();
		content << ";\n";
	}
	content <<  "}\n";
	return content.str();
}

bool hasBranching(int* parents, int n){
	bool* isParent = new bool[n+1];
	for(int i=0; i<n; i++){
		isParent[i]=false;
	}
	for(int i=0; i<n; i++){
		if(isParent[parents[i]]==true){
			return true;
		}
		isParent[parents[i]] = true;
	}
	return false;
}


//int** getDataMatrix(int n, int m, string fileName){
//
//	//printf("%d rows, %d columns\n", n ,m);
//    int** dataMatrix = init_intMatrix(n, m, -1);
//
//
//    ifstream in(fileName);
//
//    if (!in) {
//    	cout << "2 Cannot open file " << fileName << "\n";
//      cout << fileName;
//      cout << "\n";
//      return NULL;
//    }
//
//   // cout << "file opened\n";
//    for (int i = 0; i < n; i++) {
//    	//cout << i << "\n";
//        for (int j = 0; j < m; j++) {
//            in >> dataMatrix[i][j];
//        }
//    }
//
//    in.close();
//    print_intMatrix(dataMatrix, n, m, ' ');
//    //printf("data matrix read and transposed.\n");
//    return transposeMatrix(dataMatrix, n, m);
//}
