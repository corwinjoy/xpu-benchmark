// Format number with commas

#ifndef PI_GPU_NUMBER_WITH_COMMAS_H
#define PI_GPU_NUMBER_WITH_COMMAS_H


#include <iomanip>
#include <locale>

template<class T>
std::string numberFormatWithCommas(T value){
    struct Numpunct: public std::numpunct<char>{
    protected:
        virtual char do_thousands_sep() const{return ',';}
        virtual std::string do_grouping() const{return "\03";}
    };
    std::stringstream ss;
    ss.imbue({std::locale(), new Numpunct});
    ss << std::setprecision(2) << std::fixed << value;
    return ss.str();
}

#endif //PI_GPU_NUMBER_WITH_COMMAS_H
