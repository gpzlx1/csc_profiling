#include <torch/script.h>
#include <torch/custom_class.h>
#include <string>
#include <vector>


struct MyObject : torch::CustomClassHolder {

    MyObject(){};
    MyObject(torch::Tensor self, torch::Tensor offset)
      : _self(self), _offset(offset) {};

    void compute() {
        _result = _self + _offset;
    };

    std::vector<torch::Tensor> ret_two(){
      return {_self, _offset};
    }


  torch::Tensor _self;
  torch::Tensor _offset;
  torch::Tensor _result;

};


torch::Tensor my_add(torch::Tensor self, torch::Tensor offset){
    return self + offset;
};

std::vector<torch::Tensor> my_flag(torch::Tensor self, torch::Tensor offset, int64_t flag){
    switch (flag)
    {
    case 1:
      return {self + 1,};
      break;
    
    case 2:
      return {self + 1, offset + 2};
      break;
    
    default:
      return {self + 1, offset + 2, self + 3};
    }
    return {};
};


template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }
};

TORCH_LIBRARY(my_ops, m) {
  m.def("my_add", &my_add);
  m.def("my_flag", &my_flag);
}


TORCH_LIBRARY(my_classes, m) {
  m.class_<MyObject>("MyObject")
   .def(torch::init<torch::Tensor, torch::Tensor>())
   .def("compute", &MyObject::compute)
   .def("ret2", &MyObject::ret_two)
   .def("get", [](const c10::intrusive_ptr<MyObject>& self) {
      return self->_result;
    })
  ;

  m.class_<MyStackClass<std::string>>("MyStackClass")
    .def(torch::init<std::vector<std::string>>())
    .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
      return self->stack_.back();
    })
    .def("push", &MyStackClass<std::string>::push)
    .def("pop", &MyStackClass<std::string>::pop)
    .def("clone", &MyStackClass<std::string>::clone)
    .def("merge", &MyStackClass<std::string>::merge)
  ;
}

//static auto registry = torch::RegisterOperators(
//    "my_ops::my_add(Tensor self, Tensor offset) -> Tensor ret",
//    &my_add);




