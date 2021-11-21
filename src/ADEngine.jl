module ADEngine

import Base: +, *, -, /, ^

export Node, backward, relu


mutable struct Node
    data::Number
    childeren::Set{Node}
    partial_derivatives::Dict
    grad::Number
end


Node(data::Number, childeren::Set{Node}) = Node(data, childeren, Dict(), 0.0);
Node(data::Number, childeren::Set{Node}, partial_derivatives::Dict) = begin
    Node(data, childeren, partial_derivatives, 0.0)
end;
Node(data::Number) = Node(data, Set{Node}());


############## Add
function +(self::Node, other::Node)::Node
    Node(
        self.data + other.data,
        Set{Node}((self, other)),
        Dict(self => 1.0, other => 1.0)
    )
end;
+(a::Number, self::Node)::Node = (Node(a) + self);
+(self::Node, a::Number)::Node = (a + self);

############## Sub
-(self::Node, other::Node)::Node =  self + (-1. * other)
-(a::Number, self::Node)::Node = (Node(a) - self);
-(self::Node, a::Number)::Node = (self - Node(a));

############# Mul
function *(self::Node, other::Node)::Node
    Node(
        self.data * other.data,
        Set{Node}((self, other)),
        Dict(self => other.data, other => self.data)
    )
end;
*(a::Number, self::Node)::Node = (Node(a) * self);
*(self::Node, a::Number)::Node = (a * self);

############# Inv
function inv(self::Node)::Node
    @assert self.data != 0
    Node(
        1. / self.data,
        Set{Node}((self,)),
        Dict(self => (-1.) * self.data ^ (-2.0))
    )
end

############# Pow
function ^(base::Node, exp::Real)::Node
    Node(
        base.data ^ exp,
        Set{Node}((base,)),
        Dict(base => exp * base.data ^ (exp-1.0))
    )
end

############# Div
/(self::Node, other::Node)::Node = self * inv(other)
/(a::Number, self::Node)::Node = Node(a) / self
/(self::Node, a::Number)::Node = self / Node(a)


############# Ops
function relu(self::Node)::Node
    Node( 
        max(0, self.data),
        Set{Node}((self,)),
        Dict(self => (self.data <= 0 ? 0.0 : 1.0)),
    )
end;


function topo(node::Node)::Vector{Node}
    function dfs(node::Node, visited, list)
        if node in visited
            return list
        end
        push!(visited, node)
        for child in node.childeren
            dfs(child, visited, list)
        end
        return push!(list, node)
    end
    return reverse(dfs(node, Set{Node}(), Vector{Node}()))
end;


function backward(node::Node)
    list = topo(node)
    list .|> (x -> x.grad = 0)
    node.grad = 1.0
    for node in list
        df_dchild = node.partial_derivatives
        for child in node.childeren
            child.grad += get(df_dchild, child, nothing) * node.grad
        end
    end
end;


end # module