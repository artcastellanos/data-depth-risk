import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group, multivariate_normal, multivariate_t

def moonrising(n):
    moon = np.empty((n,2))
    moon[:,0] = np.random.uniform(-1,1,n)
    moon[:,1] = np.random.uniform(1.5*(1-moon[:,0]**2),2*(1-moon[:,0]**2))
    # for i in range(n):
    #     moon[i,1] = np.random.uniform(1.5*(1-moon[i,0]**2),2*(1-moon[i,0]**2))
    return moon

def two_moons(n):
    moon1 = moonrising(n)
    moon1[:,0] += 1
    moon2 = -moonrising(n)
    moon2[:,0] += -1
    return np.concatenate((moon1,moon2))

def twogaussians(n,d):
    gaussian1 = np.random.standard_normal(size=(n, d)) + 3.5
    gaussian2 = np.random.standard_normal(size=(n, d)) - 3.5
    return np.concatenate((gaussian1,gaussian2))

def unbalanced_twogaussians(r,n,d):
    n1 = int(2*r*n)
    n2 = 2*n - n1
    gaussian1 = np.random.standard_normal(size=(n1, d)) + 3.5
    gaussian2 = np.random.standard_normal(size=(n2, d)) - 3.5
    return np.concatenate((gaussian1,gaussian2))

def three_gaussians(n_tot,d):
    n = n_tot//3
    gaussian1 = np.random.standard_normal(size=(n+(n_tot%3), d)) 
    gaussian2 = np.random.standard_normal(size=(n, d)) + 7
    gaussian3 = np.random.standard_normal(size=(n, d)) - 7
    return np.concatenate((gaussian1,gaussian2,gaussian3))

def five_gaussians(n_tot,d):
    n = n_tot//5
    gaussian1 = np.random.standard_normal(size=(n+(n_tot%5), d)) 
    gaussian2 = np.random.standard_normal(size=(n, d)) + 7
    gaussian3 = np.random.standard_normal(size=(n, d)) - 7
    gaussian4 = np.random.standard_normal(size=(n, d)) + 14
    gaussian5 = np.random.standard_normal(size=(n, d)) - 14
    return np.concatenate((gaussian1,gaussian2,gaussian3,gaussian4,gaussian5))

def unbalanced_three_gaussians(n_tot,d):
    n = n_tot//4
    gaussian1 = np.random.standard_normal(size=(n, d)) 
    gaussian2 = np.random.standard_normal(size=(2*n, d)) + 7
    gaussian3 = np.random.standard_normal(size=(n, d)) - 7
    return np.concatenate((gaussian1,gaussian2,gaussian3))

def power_gaussians(n_tot,d,amplitude=3.5):
    n = n_tot//d
    mean = np.zeros(d)
    mean[0] = amplitude
    gaussians = mean + np.random.standard_normal(size=(n+(n_tot%d), d)) 
    for i in range(1,d):
        mean = np.zeros(d)
        mean[i] = amplitude    
        current_gaussian = mean + np.random.standard_normal(size=(n, d)) 
        gaussians = np.concatenate((gaussians,current_gaussian))
    return gaussians

def contaminated_gaussian(n,d,contamination_ratio):
    gaussian = np.random.standard_normal(size=(n, d))
    n_contamination = int(n*contamination_ratio)
    gaussian[-n_contamination:,:] = np.random.standard_normal(size=(n_contamination, d)) + 3.5
    return gaussian

def circle(n,low=0,high=1):
    rho = np.sqrt(np.random.uniform(low=low,high=high,size=n))
    # print(rho)
    theta = np.random.uniform(low=0,high=2*np.pi,size=n)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.column_stack((x,y))

def two_donuts(n):
    donut1 = circle(3*n,1,4)
    donut1[:,0] += 2
    donut2 = circle(n,0.25,1)
    donut2[:,0] -= 1
    return np.concatenate((donut1,donut2))

def gen_rand_covmat(d,sigmas=None):
    P = ortho_group.rvs(dim=d)
    if sigmas == None:
        sigmas = np.random.standard_normal(size=(d, 1))
    D = np.diagflat(sigmas**2)
    covmat = P @ D @ P.T
    return covmat

def student(n=100,d=2,mat=None,df=2):
    rv = multivariate_t(loc=np.zeros(d),shape=mat,df=df)
    samples = rv.rvs(size=n)
    return samples

def two_students(n,d,df=None):
    student1 = student(n,d,df) + 3.5
    student2 = student(n,d,df) - 3.5
    return np.concatenate((student1,student2))

def distribute(distrib='normal',n=100,d=2,covmat=None,r=0.7,low=1,high=4,df=None):
    if distrib == 'uniform':
        X = X = np.random.rand(n,d) 
    if distrib == 'normal':
        X = np.random.standard_normal(size=(n, d))
    if distrib == 'covmat':
        X = np.random.multivariate_normal(mean=np.zeros(d),cov=covmat,size=n)
    if distrib == 'two_gaussians':
        X = twogaussians(n//2,d)
    if distrib == 'unbalanced_twogaussians':
        X = unbalanced_twogaussians(r,n//2,d)
    if distrib == 'three_gaussians':
        X = three_gaussians(n,d)
    if distrib == 'five_gaussians':
        X = five_gaussians(n, d)
    if distrib == 'moon':
        X = moonrising(n)
    if distrib == 'two_moons':
        X = two_moons(n//2)  
    if distrib == 'donut':
        X = circle(n,low,high)
    if distrib == 'two_donuts':
        X = two_donuts(n//4)
    if distrib == 'power_gaussians':
        X = power_gaussians(n,d)
    if distrib == 'student':
        X = student(n,d,df)
    if distrib == 'two_students':
        X = two_students(n//2,d,df)
    return X

def get_densities(distrib,X,covmat=None,amplitude=3.5,df=None):
    n,d = X.shape
    if distrib == 'normal':
        gauss = multivariate_normal(mean=np.zeros(d), cov=np.identity(d))
        densities = gauss.pdf(X)
    if distrib == 'two_gaussians':
        gauss1 = multivariate_normal(mean=np.zeros(d)+3.5, cov=np.identity(d))
        densities1 = gauss1.pdf(X[:n//2])
        gauss2 = multivariate_normal(mean=np.zeros(d)-3.5, cov=np.identity(d))
        densities2 = gauss2.pdf(X[n//2:])
        densities = np.concatenate((densities1,densities2))
        # densities = densities.reshape(-1,1)
    if distrib == 'three_gaussians':
        n_third = n//3
        X1 = X[:n_third+(n%3)]
        X2 = X[(n_third+(n%3)):(2*n_third+(n%3))]
        X3 = X[(2*n_third+(n%3)):]
        gauss1 = multivariate_normal(mean=np.zeros(d), cov=np.identity(d))
        densities1 = gauss1.pdf(X1)
        gauss2 = multivariate_normal(mean=np.zeros(d)+7, cov=np.identity(d))
        densities2 = gauss2.pdf(X2)
        gauss3 = multivariate_normal(mean=np.zeros(d)-7, cov=np.identity(d))
        densities3 = gauss3.pdf(X3)
        densities = np.concatenate((densities1,densities2,densities3))
    if distrib == 'covmat':
        gauss = multivariate_normal(mean=np.zeros(d), cov=covmat)
        densities = gauss.pdf(X)
    if distrib == 'power_gaussians':
        n_div = n//d
        # init first gaussians
        mean = np.zeros(d)
        mean[0] = amplitude
        current_X = X[:n_div+(n%d)]    
        gaussian = multivariate_normal(mean=mean, cov=np.identity(d))
        densities = gaussian.pdf(current_X)
        # loop for the rest
        for i in range(1,d):
            current_X = X[i*n_div+(n%d):(i+1)*n_div+(n%d)]
            mean = np.zeros(d)
            mean[i] = amplitude    
            current_gaussian = multivariate_normal(mean=mean, cov=np.identity(d))
            current_densities = current_gaussian.pdf(current_X)
            densities = np.concatenate((densities,current_densities))
    if distrib == 'student':
        stud = multivariate_t(loc=np.zeros(d),shape=covmat,df=df)
        densities = stud.pdf(X)
    if distrib == 'two_students':
        stud1 = multivariate_t(loc=np.zeros(d)+3.5,df=df)
        densities1 = stud1.pdf(X[:n//2])
        stud2 = multivariate_t(loc=np.zeros(d)-3.5,df=df)
        densities2 = stud2.pdf(X[n//2:])
        densities = np.concatenate((densities1,densities2))
    return densities






if __name__=='__main__':
    n= 1000
    d = 2
    seed_num = 42
    np.random.seed(seed_num)
    # data = moonrising(n)
    # data = two_moons(n)
    # data = np.random.standard_normal(size=(n, d))
    # data = np.random.standard_t(5,size=(n, d))
    # data = np.random.standard_cauchy(size=(n, d))
    # data = np.random.uniform(-1,1,(n,d))
    # data = np.random.exponential(size=(n,d))
    # data = twogaussians(n,d)
    # data = circle(n,1,4)
    # data = circle(n,0.25,1)
    # data = two_donuts(n)
    # data = three_gaussians(n,d)
    data = np.random.multivariate_normal(mean=[0,0],cov=[[1,1],[1,4]],size=n)



    print(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()
