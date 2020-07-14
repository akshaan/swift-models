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
    associatedtype Element = TensorGroup
    var componentTypes: [TensorDataType] { get }
    var shapes: [TensorShape?] { get }
    var names: [String]? { get }
    var queueRef: ResourceHandle { get }
    
    func enqueue(vals: Element, name: String?)
    func enqueueMany(vals: Element, name: String?)
    func dequeue(name: String?) -> Element
    func dequeueMany(n: Int, name: String?) -> Element
    func dequeueUpTo(n: Int, name: String?) -> Element
    func close(cancelPendingEnqueues: Bool, name: String?)
    func isClosed(name: String?) -> Bool
    func size(name: String?) -> Int
}
