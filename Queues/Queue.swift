// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

protocol Queue {
    var componentTypes: [TensorDataType] { get }
    var shapes: [TensorShape?] { get }
    var names: [String]? { get }
    var queueRef: ResourceHandle { get }
    
    func enqueue<T: TensorArrayProtocol>(vals: T, name: String?)
    func enqueueMany<T: TensorArrayProtocol>(vals: T, name: String?)
    func dequeue<T: TensorGroup>(name: String?) -> T
    func dequeueMany<T: TensorGroup>(n: Tensor<Int32>, name: String?) -> T
    func dequeueUpTo<T: TensorGroup>(n: Tensor<Int32>, name: String?) -> T
    func close(cancelPendingEnqueues: Bool, name: String?)
    func isClosed(name: String?) -> Tensor<Bool>
    func size(name: String?) -> Tensor<Int32>
}
