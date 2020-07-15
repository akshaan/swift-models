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

import Foundation
import TensorFlow

class PaddingFIFOQueue: Queue {
    let componentTypes: [TensorDataType]
    let shapes: [TensorShape?]
    let names: [String]?
    let queueRef: ResourceHandle
    let capacity: Int
    let sharedName: String
    
    init(
        capacity: Int,
        dtypes: [TensorDataType],
        shapes:  [TensorShape?],
        names: [String]? = nil,
        sharedName: String? = nil) {
        self.capacity = capacity
        self.componentTypes = dtypes
        self.shapes = shapes
        self.names = names
        let sharedNameOrDefault = sharedName ?? UUID().uuidString
        self.sharedName = sharedNameOrDefault
        self.queueRef = _Raw.paddingFIFOQueueV2(componentTypes: dtypes, shapes: shapes, container: "default", sharedName: sharedNameOrDefault)
    }
    
    public func enqueue<T>(vals: T, name: String? = nil) where T: TensorArrayProtocol {
        _Raw.queueEnqueueV2(handle: self.queueRef, components: vals)
    }
    
    public func enqueueMany<T>(vals: T, name: String?) where T : TensorArrayProtocol {
        _Raw.queueEnqueueManyV2(handle: self.queueRef, components: vals)
    }
    
    public func dequeue<T>(name: String?) -> T where T : TensorGroup {
        _Raw.queueDequeueV2(handle: self.queueRef)
    }
    
    public func dequeueMany<T>(n: Tensor<Int32>, name: String?) -> T where T : TensorGroup {
        _Raw.queueDequeueManyV2(handle: self.queueRef, n: n)
    }
    
    public func dequeueUpTo<T>(n: Tensor<Int32>, name: String?) -> T where T : TensorGroup {
        _Raw.queueDequeueUpToV2(handle: self.queueRef, n: n)
    }
    
    public func close(cancelPendingEnqueues: Bool, name: String?){
        _Raw.queueCloseV2(handle: self.queueRef, cancelPendingEnqueues: cancelPendingEnqueues)
    }
    
    public func isClosed(name: String?) -> Tensor<Bool> {
        _Raw.queueIsClosedV2(handle: self.queueRef)
    }
    
    public func size(name: String?) -> Tensor<Int32> {
        _Raw.queueSizeV2(handle: self.queueRef)
    }
}
