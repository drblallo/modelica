#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/matching/KhanAdjacentAlgorithm.hpp"
#include "marco/matching/MatchedEquationLookup.hpp"
#include "marco/matching/SccLookup.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
namespace marco
{
	/**
	 * A reference to a scalar equation, that is a reference to a scalar equation
	 * and a set of indicies that indicate which scalar equation we are reffering
	 * to
	 */
	class SingleEquationReference
	{
		public:
		SingleEquationReference(
				const IndexesOfEquation& vertex, llvm::SmallVector<size_t, 3> indexes)
				: vertex(&vertex), indexes(std::move(indexes))
		{
		}

		SingleEquationReference() = default;

		[[nodiscard]] const auto& getCollapsedVertex() const
		{
			return vertex->getContent();
		}

		[[nodiscard]] auto getIndexes() const { return indexes; }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		const IndexesOfEquation* vertex{ nullptr };
		llvm::SmallVector<size_t, 3> indexes;
	};

	class SVarDependencyGraph
	{
		public:
		using GraphImp = boost::adjacency_list<
				boost::vecS,
				boost::vecS,
				boost::directedS,
				SingleEquationReference>;

		using VertexIndex =
				boost::property_map<GraphImp, boost::vertex_index_t>::type::value_type;

		using VertexDesc = boost::graph_traits<GraphImp>::vertex_descriptor;

		using VVarScc = Scc<VVarDependencyGraph>;

		using LookUp = std::map<const IndexesOfEquation*, std::map<size_t, size_t>>;

		SVarDependencyGraph(
				const VVarDependencyGraph& collapsedGraph, const VVarScc& scc);

		[[nodiscard]] const VVarScc& getScc() const { return scc; }
		[[nodiscard]] size_t count() const { return boost::num_vertices(graph); }
		[[nodiscard]] const VVarDependencyGraph::GraphImp& collImpl() const
		{
			return collapsedGraph.getImpl();
		}
		void dumpGraph(llvm::raw_ostream& OS) const;
		template<typename OnSched, typename OnGroupDone>
		void topoOrder(OnSched onSched, OnGroupDone onGroupDOne) const
		{
			khanAdjacentAlgorithm(graph, std::move(onSched), std::move(onGroupDOne));
		}

		[[nodiscard]] auto operator[](size_t index) const { return graph[index]; }

		private:
		void insertEdge(
				const LookUp& lookup, const VVarDependencyGraph::EdgeDesc& edge);
		void insertNode(LookUp& lookUp, size_t vertexIndex);
		void insertEdges(const LookUp& lookup, size_t vertex);
		const VVarScc& scc;
		const VVarDependencyGraph& collapsedGraph;
		GraphImp graph;
	};
}	 // namespace marco
