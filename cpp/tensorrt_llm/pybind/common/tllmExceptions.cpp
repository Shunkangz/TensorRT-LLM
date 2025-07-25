/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tllmExceptions.h"

namespace py = pybind11;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::pybind::common
{

void bindRequestSpecificException(py::module_& m)
{
    // Bind the RequestErrorCode enum
    py::enum_<tc::RequestErrorCode>(m, "RequestErrorCode")
        .value("UNKNOWN_ERROR", tc::RequestErrorCode::UNKNOWN_ERROR)
        .value("NETWORK_ERROR", tc::RequestErrorCode::NETWORK_ERROR)
        .export_values();

    // Create the RequestSpecificException Python exception class
    static PyObject* request_specific_exc
        = PyErr_NewException("tensorrt_llm.RequestSpecificException", nullptr, nullptr);

    m.add_object("RequestSpecificException", py::handle(request_specific_exc));

    // Register exception translator to convert C++ exceptions to Python
    py::register_exception_translator(
        [](std::exception_ptr p)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (const tc::RequestSpecificException& e)
            {
                // Create a Python exception with the request ID and error code information
                py::object py_exc = py::cast(e);
                py::object request_id = py::cast(e.getRequestId());
                py::object error_code = py::cast(static_cast<uint32_t>(e.getErrorCode()));

                // Set additional attributes on the exception
                py_exc.attr("request_id") = request_id;
                py_exc.attr("error_code") = error_code;
                py_exc.attr("error_code_enum") = py::cast(e.getErrorCode());

                PyErr_SetObject(request_specific_exc, py_exc.ptr());
            }
        });

    // Bind the C++ RequestSpecificException class to Python
    py::class_<tc::RequestSpecificException, std::runtime_error>(m, "RequestSpecificExceptionCpp")
        .def(py::init<char const*, std::size_t, char const*, uint64_t, tc::RequestErrorCode>(), py::arg("file"),
            py::arg("line"), py::arg("msg"), py::arg("requestID"), py::arg("errorCode"))
        .def("getRequestId", &tc::RequestSpecificException::getRequestId)
        .def("getErrorCode", &tc::RequestSpecificException::getErrorCode)
        .def_readonly("request_id", &tc::RequestSpecificException::getRequestId)
        .def_readonly("error_code", &tc::RequestSpecificException::getErrorCode)
        .def("__str__", [](tc::RequestSpecificException const& e) { return std::string(e.what()); })
        .def("__repr__",
            [](tc::RequestSpecificException const& e)
            {
                return "RequestSpecificException(request_id=" + std::to_string(e.getRequestId())
                    + ", error_code=" + std::to_string(static_cast<uint32_t>(e.getErrorCode())) + ")";
            });

    // Add convenience functions for creating exceptions from Python
    m.def(
        "create_request_exception",
        [](uint64_t request_id, tc::RequestErrorCode error_code, std::string const& message)
        { return tc::RequestSpecificException(__FILE__, __LINE__, message.c_str(), request_id, error_code); },
        py::arg("request_id"), py::arg("error_code"), py::arg("message"),
        "Create a RequestSpecificException from Python");

    m.def(
        "create_network_error",
        [](uint64_t request_id, std::string const& message)
        {
            return tc::RequestSpecificException(
                __FILE__, __LINE__, message.c_str(), request_id, tc::RequestErrorCode::NETWORK_ERROR);
        },
        py::arg("request_id"), py::arg("message"), "Create a network error exception");
}

} // namespace tensorrt_llm::pybind::common
