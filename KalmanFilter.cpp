// Implementation file of the Kalman filter class

#include <iostream>
#include<tuple>
#include<string>
#include<fstream>
#include<vector>
#include<Eigen/Dense>
#include "KalmanFilter.h"

using namespace Eigen;
using namespace std;

// edit this later
KalmanFilter::KalmanFilter()
{    
}
        // Overloaded constructor
        // -  takes A,B,C, Q, R, P0, x0 matrices/vectors and assigns them to private variables
        //
        // -  the input argument "maxMeasurements" is used to initialize zero matrices 
        // "estimatesAposteriori","estimatesApriori", "covarianceAposteriori", "covarianceApriori"
        // "gainMatrices", and "errors" that are used to store appropriate quantities during the propagation 
        // of the Kalman filter equations
        // 
        // - the private variable "k" is set to zero. This variable is used to track the current iteration 
        // of the Kalman filter.
KalmanFilter::KalmanFilter(MatrixXd Ainput, MatrixXd Binput, MatrixXd Cinput, 
                           MatrixXd Qinput, MatrixXd Rinput, MatrixXd P0input, 
						   MatrixXd x0input, unsigned int maxSimulationSamples)
{
    k=0;
	// assign the private variables
     A=Ainput;   B=Binput;      C=Cinput;     Q=Qinput;
     R=Rinput;   P0=P0input;    x0=x0input;
	// extract the appropriate dimensions
    n = A.rows();   m = B.cols(); r = C.rows();

	// this matrix is used to store a posteriori estimates column wise
	estimatesAposteriori.resize(n, maxSimulationSamples); 
    estimatesAposteriori.setZero();
    estimatesAposteriori.col(0)=x0;
    
	// this matrix is used to store a priori estimates column wise
    estimatesApriori.resize(n, maxSimulationSamples); 
    estimatesApriori.setZero();
    
	// this matrix is used to store the a posteriori covariance matrices next to each other
    covarianceAposteriori.resize(n,n*maxSimulationSamples);
    covarianceAposteriori.setZero();
    covarianceAposteriori(all,seq(0,n-1))=P0;

	// this matrix is used to store the a priori covariance matrices next to each other
    covarianceApriori.resize(n,n*maxSimulationSamples);
    covarianceApriori.setZero();
    
	// this matrix is used to store the Kalman gain matrices next to each other
    gainMatrices.resize(n,r*maxSimulationSamples);
    gainMatrices.setZero();
    
	// this matrix is used to store the errors (innovations) column-wise
    errors.resize(r,maxSimulationSamples);
    errors.setZero();
}

        // this member function predicts the estimate on the basis of the external input
        // it computes the a priori estimate
        // it computes the a priori covariance matrix
void KalmanFilter::predictEstimate(MatrixXd externalInput)
{
    // keep in mind that initially, k=0
	estimatesApriori.col(k)=A*estimatesAposteriori.col(k)+B*externalInput;
    covarianceApriori(all,seq(k*n,(k+1)*n-1))=A*covarianceAposteriori(all,seq(k*n,(k+1)*n-1))*(A.transpose())+Q;
    // increment the time step
    k++; 
    
}
       
	    // this member function updates the estimate on the basis of the measurement 
        // it computes the Kalman filter gain matrix
        // it computes the a posteriori estimate 
        // it computes the a posteriori covariance matrix

void KalmanFilter::updateEstimate(MatrixXd measurement)
{
    // initially, the value of k will be 1, once this function is called
	// this is because predictEstimate() is called before this function 
	// and predict estimate increments the value of k

	// this matrix is used to compute the Kalman gain
	MatrixXd Sk;
    Sk.resize(r,r);
    Sk=R+C*covarianceApriori(all,seq((k-1)*n,k*n-1))*(C.transpose());
    Sk=Sk.inverse();
	// gain matrices 
	gainMatrices(all,seq((k-1)*r,k*r-1))=covarianceApriori(all,seq((k-1)*n,k*n-1))*(C.transpose())*Sk;
    // compute the error - innovation 
	errors.col(k-1)=measurement-C*estimatesApriori.col(k-1);
    // compute the a posteriori estimate, remember that for k=0, the corresponding column is x0 - initial guess
	estimatesAposteriori.col(k)=estimatesApriori.col(k-1)+gainMatrices(all,seq((k-1)*r,k*r-1))*errors.col(k-1);

   MatrixXd In;
   In= MatrixXd::Identity(n,n);
   MatrixXd IminusKC;
   IminusKC.resize(n,n);
   IminusKC=In-gainMatrices(all,seq((k-1)*r,k*r-1))*C;  // I-KC
   
   // update the a posteriori covariance matrix
   covarianceAposteriori(all,seq(k*n,(k+1)*n-1))
   =IminusKC*covarianceApriori(all,seq((k-1)*n,k*n-1))*(IminusKC.transpose())
   +gainMatrices(all,seq((k-1)*r,k*r-1))*R*(gainMatrices(all,seq((k-1)*r,k*r-1)).transpose());
}

        // this member function is used to load the measurement data from the external CSV file 
        // the values are stored in the output matrix 
        // MatrixXd is an Eigen typdef for Matrix<double, Dynamic, Dynamic>

MatrixXd KalmanFilter::openData(string fileToOpen)
{

	// the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
	// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
	// NOTE THAT THIS FUNCTION IS CALLED BY THE FUNCTION: SimulateSystem::openFromFile(std::string Afile, std::string Bfile, std::string Cfile, std::string x0File, std::string inputSequenceFile)
	
	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format



	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//	  d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
	vector<double> matrixEntries;

	// in this object we store the data from the matrix
	ifstream matrixDataFile(fileToOpen);

	// this variable is used to store the row of the matrix that contains commas 
	string matrixRowString;

	// this variable is used to store the matrix entry;
	string matrixEntry;

	// this variable is used to track the number of rows
	int matrixRowNumber = 0;


	while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

		while (getline(matrixRowStringStream, matrixEntry,',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
			}
		matrixRowNumber++; //update the column numbers
	}

	// here we convert the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Map<Matrix<double, Dynamic, Dynamic, RowMajor>> (matrixEntries.data(), 
															matrixRowNumber, matrixEntries.size() / matrixRowNumber);

}


 // this member function saves the stored date in the corresponding CSV files

void KalmanFilter::saveData(string estimatesAposterioriFile, string estimatesAprioriFile, 
							string covarianceAposterioriFile, string covarianceAprioriFile, 
							string gainMatricesFile, string errorsFile) const
{
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
	
	ofstream file1(estimatesAposterioriFile);
	if (file1.is_open())
	{
		file1 << estimatesAposteriori.format(CSVFormat);
		
		file1.close();
	}

	ofstream file2(estimatesAprioriFile);
	if (file2.is_open())
	{
		file2 << estimatesApriori.format(CSVFormat);
		file2.close();
	}
	
	ofstream file3(covarianceAposterioriFile);
	if (file3.is_open())
	{
		file3 << covarianceAposteriori.format(CSVFormat);
		file3.close();
	}

	ofstream file4(covarianceAprioriFile);
	if (file4.is_open())
	{
		file4 << covarianceApriori.format(CSVFormat);
		file4.close();
	}

	ofstream file5(gainMatricesFile);
	if (file5.is_open())
	{
		file5 << gainMatrices.format(CSVFormat);
		file5.close();
	}

	ofstream file6(errorsFile);
	if (file6.is_open())
	{
		file6 << errors.format(CSVFormat);
		file6.close();
	}

	
}
