#include <iostream>
#include <algorithm>
#include <iterator>
#include <valarray>
#include <list>

// #include <eigen3/Eigen/Dense>


/**
 * @author   Markus Straubinger
 *           markus.straubinger@uni-bayreuth.de
 *
 * @brief    This class implements multi-dimensional minimization by the downhill simplex
 *           search method of Nelder and Mead (1965)
 *           The implementation follows the algorithm presented on wikipedia
 *           https://de.wikipedia.org/wiki/Downhill-Simplex-Verfahren
 *
 * @tparam F objective functor
 *           needs to be compatible with <P> : point type
 *           evaluates a given point
 *           return value must be comparable
 *           lesser return values are considered better
 * @tparam S simplex type that holds the individual points
 *           needs to implement random access iterator
 *           and value_type
 *           example: std::vector
 * @tparam P point type
 *           needs to implement addition and multiplication with a scalar (double)
 *
 * @TODO     there are quite some performance optimizations possible
 *           for example you can cache the function evaluations for the simplex points
 */
template<typename F, typename S, typename P = typename S::value_type>
class NelderMead
{
public:
    typedef F functor_type;
    typedef S simplex_type;
    typedef P point_type;

    NelderMead(functor_type &f, simplex_type &s) : objective_functor(f), simplex(s) {}


    /**
     * @brief  this method will perform the Nelder-Mead algorithm until
     *         either convergence is achieved or a maximum number of
     *         iterations is reached
     *
     * @TODO   implement convergence criteria, maybe let user give some hints
     */
    void find_min()
    {
        for(std::size_t i = 0; i < 50; i++)
        {
            step();
        }
    }

    /**
     * @brief  performs a single step of the Nelder-Mead algorithm
     */
    void step()
    {
        const std::size_t N = simplex.size() - 1;

        /*
         * Schritt 2:
         *
         * sortiere die Punkte nach dem Wert der Zielfunktion {\displaystyle f} f, so dass
         * {\displaystyle x_{0}} x_{0} der beste, {\displaystyle x_{N-1}} x_{N-1}
         * der zweitschlechteste und {\displaystyle x_{N}} x_{N} der schlechteste ist
         */
        std::sort(std::begin(simplex), std::end(simplex),
                  [this](point_type a, point_type b) { return objective_functor(a) < objective_functor(b); });

        /*
         * Schritt 3:
         *https://eigen.tuxfamily.org/dox/ftv2node.png
         * bilde von allen außer dem schlechtesten Punkt den Mittelpunkt:
         * {\displaystyle m=
         * {\frac {1}{N}}\sum \nolimits _{i=0}^{N-1}x_{i}} m = \frac{1}{N} \sum\nolimits_{i = 0}^{N-1} x_i
         */
        centroid = simplex[0];
        for(std::size_t i = 1; i < N ; i++ )
        {
            centroid += simplex[i];
        }
        centroid /= N;

        /*
         * Schritt 4:
         *
         * reflektiere den schlechtesten Punkt am Mittelpunkt:
         * {\displaystyle r=(1+\alpha )m-\alpha x_{N}} r = (1+\alpha)m - \alpha x_N
         */
        point_type r = (1 + reflexion_coeff) * centroid - reflexion_coeff * simplex[N];

        /*
         * Schritt 5:
         *
         * wenn {\displaystyle r} r besser ist als {\displaystyle x_{0}} x_{0}:
         * bestimme den expandierten Punkt
         * {\displaystyle e=(1+\gamma )m-\gamma x_{N}} e = (1+\gamma)m - \gamma x_N,
         * ersetze {\displaystyle x_{N}} x_{N}
         * durch den besseren der beiden Punkte {\displaystyle e,r} e, r und gehe zu Schritt 2
         */
        if(objective_functor(r) < objective_functor(simplex[0]))
        {
            point_type e = (1 + expansion_coeff) * centroid - expansion_coeff *  simplex[N];
            simplex[N] = objective_functor(r) < objective_functor(e) ? r : e;
            return;
        }

        /*
         * Schritt 6:
         *
         * wenn {\displaystyle r} r besser ist als der zweitschlechteste Punkt {\displaystyle x_{N-1}} x_{N-1}:
         * ersetze {\displaystyle x_{N}} x_{N} durch {\displaystyle r} r und gehe zu Schritt 2
         */
        if(objective_functor(r) < objective_functor(simplex[N-1]))
        {
            simplex[N] = r;
            return;
        }

        /*
         * Schritt 7:
         *
         * sei {\displaystyle h} h der bessere der beiden Punkte {\displaystyle x_{N},r} x_N, r.
         * Bestimme den kontrahierten Punkt {\displaystyle c=\beta m+(1-\beta )h} c = \beta m + (1-\beta) h
         */
        point_type h = objective_functor(simplex[N]) < objective_functor(r) ? simplex[N] : r;
        point_type c = contraction_coeff * centroid + (1 - contraction_coeff) * h;

        /*
         * Schritt 8:
         *
         * wenn {\displaystyle c} c besser ist als {\displaystyle x_{N}} x_{N}:
         * ersetze {\displaystyle x_{N}} x_{N} durch {\displaystyle c} c und gehe zu Schritt 2
         */
        if(objective_functor(c) < objective_functor(simplex[N]))
        {
            simplex[N] = c;
            return;
        }

        /*
         * Schritt 9:
         *
         * komprimiere den Simplex: für jedes {\displaystyle i\in \{1,\cdots ,N\}} i \in \{1,\cdots , N\}:
         * ersetze {\displaystyle x_{i}} x_{i} durch
         * {\displaystyle \sigma x_{0}+(1-\sigma )x_{i}} \sigma x_0 + (1-\sigma) x_i
         */
        for(std::size_t i = 1; i <= N; i++)
        {
            simplex[i] = shrinkage_coeff * simplex[0] + (1 - shrinkage_coeff) * simplex[i];
        }

    }

    void print_simplex()
    {
        for (auto &i: simplex)
        {
            for (std::size_t j = 0; j < i.size(); j++)
            {
                std::cout << i[j] << " ";
            }
            std::cout << "-> " << objective_functor(i) << std::endl;
        }
        std::cout << std::endl;
    }

    void print_centroid()
    {
        for (std::size_t i = 0; i < centroid.size(); i++)
        {
            std::cout << centroid[i] << " ";
        }
        std::cout << std::endl;
    }

    void print_point(point_type& p)
    {
        std::cout << "point : " ;
        for (std::size_t i = 0; i < p.size(); i++)
        {
            std::cout << p[i] << " ";
        }
        std::cout << std::endl;
    }

protected:
    simplex_type simplex;
    functor_type objective_functor;
    point_type centroid;

    double   reflexion_coeff = 1.0;
    double   expansion_coeff = 2.0;
    double contraction_coeff = 0.5;
    double   shrinkage_coeff = 0.5;
};


class EulerDist2D
{
public:
    double operator()(const std::valarray<double> &v) const
    {
        double sum = 0.0;

        for (auto &i : v)
        {
            sum += i * i;
        }

        return sum;
    }
};

/*
class EigenDist2D
{
public:
    double operator()(const Eigen::Vector2d &v) const
    {
        double sum = 0.0;

        for (std::size_t i = 0; i< v.size() ; i++)
        {
            sum += v[i] * v[i];
        }

        return sum;
    }
};
*/


int main()
{
    /*
     * Valarray Demonstration
     */

    EulerDist2D euler;

    std::valarray<double> p1{3.6, 7.4};
    std::valarray<double> p2{12.6, 9.1};
    std::valarray<double> p3{-1.3, 1.2};

    std::vector<std::valarray<double>> simplex;
    simplex.push_back(p1);
    simplex.push_back(p2);
    simplex.push_back(p3);

    NelderMead<decltype(euler), decltype(simplex)> nm(euler, simplex);

    nm.print_simplex();
    nm.find_min();
    nm.print_centroid();
    std::cout << "ENDE" << std::endl;
    std::cout << std::endl;


    /*
     * Eigen3 Demonstration
     */

    EigenDist2D eigen_euler;

    Eigen::Vector2d e1(3.6, 7.4);
    Eigen::Vector2d e2(12.6, 9.1);
    Eigen::Vector2d e3(-1.3, 1.2);

    std::vector<Eigen::Vector2d> eigen_simplex;
    eigen_simplex.push_back(e1);
    eigen_simplex.push_back(e2);
    eigen_simplex.push_back(e3);

    NelderMead<decltype(eigen_euler), decltype(eigen_simplex)> eigen_nm(eigen_euler, eigen_simplex);


    eigen_nm.print_simplex();
    eigen_nm.find_min();
    eigen_nm.print_centroid();


    return 0;
}