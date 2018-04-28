library(ggplot2)
alldata = read.csv('graphs/results.csv')

strgrapher <- function(exclude, ty, wl, title) {
	ftab = subset(subset(subset(alldata, data.type == ty), workload == wl), number.of.elements != exclude)
	tags=ftab$data.structure
	sizes=ftab$str.number.of.elements
	optimes=ftab$mean.time.per.operation.ns
	btab = data.frame(Size=sizes, Data.Structure=tags, ns.Per.Op=optimes)
	# print(btab)
	png(filename=paste('graphs/', paste(ty,wl,sep='_'), '.png', sep=''), width=600, height=600)
	posns <- c("16K", "1M", "16M")
	ggplot(btab, aes(fill=Data.Structure,y=ns.Per.Op, x=Size)) + 
		scale_x_discrete(limits=posns) +
		geom_bar(position="dodge", stat="identity") +
		labs(title=title, x="Number of Elements", y="ns Per Operation")
}

intgrapher <- function(size, ty1, ty2, wl, title) {
	ftab1 = subset(subset(subset(alldata, number.of.elements == size), data.type == ty1), workload == wl)
	ftab2 = subset(subset(subset(alldata, number.of.elements == size), data.type == ty2), workload == wl)
	png(filename=paste('graphs/', paste(ty1,ty2,wl,size,sep='_'), '.png', sep=''), width=600, height=600)
	tags = c(rep('dense', length(ftab1$mean.time.per.operation.ns)),
		 rep('sparse', length(ftab1$mean.time.per.operation.ns)))
	btab = data.frame(Data.Structure=rep(ftab1$data.structure, 2),
			  ns.Per.Operation=c(ftab1$mean.time.per.operation.ns, ftab2$mean.time.per.operation.ns),
			  workload=tags)
	ggplot(btab, aes(fill=workload, y=ns.Per.Operation, x=Data.Structure)) +
		geom_bar(position="dodge", stat="identity") +
		labs(title=title, x="Data Structure", y="ns Per Operation")
}

intgrapher(16384,     'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 16K elements')
intgrapher(16777216,  'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 16M elements')
intgrapher(268435456, 'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 256M elements')

intgrapher(16384,     'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 16K elements')
intgrapher(16777216,  'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 16M elements')
intgrapher(268435456, 'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 256M elements')

intgrapher(16384,     'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 16K elements')
intgrapher(16777216,  'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 16M elements')
intgrapher(268435456, 'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 256M elements')

strgrapher(67108864,'String', 'lookup_hit', 'Lookups in the set, UTF-8 Strings of mean length 10')
strgrapher(67108864,'String', 'lookup_miss', 'Lookups not in the set, UTF-8 Strings of mean length 10')
strgrapher(67108864,'String', 'insert_remove', 'Insert/Remove Pairs, UTF-8 Strings of mean length 10')
