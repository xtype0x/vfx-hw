#ifndef KDTREE_H
#define KDTREE_H

class KdNode;

class Kdtree
{
public:
	Kdtree(){
		root = NULL;
	}
	~Kdtree();
private:
	KdNode *root;
};

class KdNode{
public:
	KdNode();
	~KdNode();

	KdNode *left;
	KdNode *right;
};

#endif